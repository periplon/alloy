//! Embedded hybrid storage combining Tantivy (full-text) and LanceDB (vector).

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use tokio::sync::{Mutex, RwLock};

use crate::error::{Result, StorageError};
use crate::search::rrf_fusion;
use crate::storage::lance_backend::LanceBackend;
use crate::storage::tantivy_backend::TantivyBackend;
use crate::storage::{
    IndexedDocument, SearchQuery, StorageBackend, StorageSearchResult, StorageStats, VectorChunk,
};

/// Embedded hybrid storage backend.
///
/// Combines:
/// - Tantivy for BM25 full-text search
/// - LanceDB for vector similarity search
/// - RRF (Reciprocal Rank Fusion) for result combination
pub struct EmbeddedStorage {
    tantivy: TantivyBackend,
    lance: Mutex<LanceBackend>,
    documents: RwLock<HashMap<String, IndexedDocument>>,
    chunk_to_doc: RwLock<HashMap<String, String>>,
    dimension: usize,
}

impl EmbeddedStorage {
    /// Create a new embedded storage backend.
    pub async fn new(data_dir: &Path, dimension: usize) -> Result<Self> {
        std::fs::create_dir_all(data_dir).map_err(StorageError::Io)?;

        let tantivy = TantivyBackend::new(data_dir)?;
        let lance = LanceBackend::new(data_dir, dimension).await?;

        Ok(Self {
            tantivy,
            lance: Mutex::new(lance),
            documents: RwLock::new(HashMap::new()),
            chunk_to_doc: RwLock::new(HashMap::new()),
            dimension,
        })
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[async_trait]
impl StorageBackend for EmbeddedStorage {
    async fn store(&self, doc: IndexedDocument, vectors: Vec<VectorChunk>) -> Result<()> {
        // Store in Tantivy
        self.tantivy.index_document(&doc)?;
        self.tantivy.commit()?;

        // Track chunk to document mapping
        {
            let mut chunk_map = self.chunk_to_doc.write().await;
            for chunk in &vectors {
                chunk_map.insert(chunk.id.clone(), doc.id.clone());
            }
        }

        // Store vectors in LanceDB
        {
            let lance = self.lance.lock().await;
            lance.store(vectors).await?;
        }

        // Store document metadata
        {
            let mut docs = self.documents.write().await;
            docs.insert(doc.id.clone(), doc);
        }

        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<StorageSearchResult>> {
        let limit = query.limit;
        let vector_weight = query.vector_weight.clamp(0.0, 1.0);
        let text_weight = 1.0 - vector_weight;

        let mut result_lists: Vec<Vec<(String, f32)>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        // Full-text search (if weight > 0)
        let mut text_results_map: HashMap<String, (String, f32)> = HashMap::new();
        if text_weight > 0.0 {
            let text_results = self.tantivy.search(
                &query.text,
                query.source_id.as_deref(),
                limit * 2, // Get extra for better fusion
            )?;

            for result in text_results {
                text_results_map.insert(
                    result.document_id.clone(),
                    (result.snippet.clone(), result.score),
                );
            }

            let text_list: Vec<(String, f32)> = text_results_map
                .iter()
                .map(|(id, (_, score))| (id.clone(), *score))
                .collect();

            if !text_list.is_empty() {
                result_lists.push(text_list);
                weights.push(text_weight);
            }
        }

        // Vector search (if embedding provided and weight > 0)
        let mut vector_results_map: HashMap<String, (String, f32)> = HashMap::new();
        if let Some(embedding) = &query.embedding {
            if vector_weight > 0.0 {
                let lance = self.lance.lock().await;
                let vector_results = lance.search(embedding, limit * 2, None).await?;

                for result in vector_results {
                    // Map chunk to document
                    let chunk_map = self.chunk_to_doc.read().await;
                    if let Some(doc_id) = chunk_map.get(&result.chunk_id) {
                        let entry = vector_results_map
                            .entry(doc_id.clone())
                            .or_insert((result.text.clone(), 0.0));
                        // Take the max score for the document
                        if result.score > entry.1 {
                            entry.0 = result.text.clone();
                            entry.1 = result.score;
                        }
                    }
                }

                let vector_list: Vec<(String, f32)> = vector_results_map
                    .iter()
                    .map(|(id, (_, score))| (id.clone(), *score))
                    .collect();

                if !vector_list.is_empty() {
                    result_lists.push(vector_list);
                    weights.push(vector_weight);
                }
            }
        }

        // Fuse results using RRF
        let fused = rrf_fusion(&result_lists, 60.0, &weights);

        // Build final results
        let mut results = Vec::new();
        let docs = self.documents.read().await;

        for (doc_id, score) in fused.into_iter().take(limit) {
            // Get text from best available source
            let text = vector_results_map
                .get(&doc_id)
                .map(|(t, _)| t.clone())
                .or_else(|| text_results_map.get(&doc_id).map(|(t, _)| t.clone()))
                .unwrap_or_default();

            // Get individual scores
            let text_score = text_results_map.get(&doc_id).map(|(_, s)| *s);
            let vector_score = vector_results_map.get(&doc_id).map(|(_, s)| *s);

            // Apply source filter if specified
            if let Some(source_id) = &query.source_id {
                if let Some(doc) = docs.get(&doc_id) {
                    if doc.source_id != *source_id {
                        continue;
                    }
                }
            }

            results.push(StorageSearchResult {
                document_id: doc_id.clone(),
                chunk_id: doc_id.clone(), // For embedded, chunk_id = doc_id for simplicity
                text,
                score,
                vector_score,
                text_score,
            });
        }

        Ok(results)
    }

    async fn remove(&self, doc_id: &str) -> Result<()> {
        // Remove from Tantivy
        self.tantivy.remove(doc_id)?;
        self.tantivy.commit()?;

        // Remove from LanceDB
        {
            let lance = self.lance.lock().await;
            lance.remove_document(doc_id).await?;
        }

        // Remove from document map
        {
            let mut docs = self.documents.write().await;
            docs.remove(doc_id);
        }

        // Clean up chunk mappings
        {
            let mut chunk_map = self.chunk_to_doc.write().await;
            chunk_map.retain(|_, v| v != doc_id);
        }

        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Option<IndexedDocument>> {
        let docs = self.documents.read().await;
        Ok(docs.get(doc_id).cloned())
    }

    async fn stats(&self) -> Result<StorageStats> {
        let document_count = self.tantivy.document_count()?;

        let chunk_count = {
            let lance = self.lance.lock().await;
            lance.chunk_count().await?
        };

        // Estimate storage size (not precise, but indicative)
        let storage_bytes = (document_count * 1024 + chunk_count * 512) as u64;

        Ok(StorageStats {
            document_count,
            chunk_count,
            storage_bytes,
        })
    }

    async fn get_all_chunks_for_clustering(
        &self,
        source_id: Option<&str>,
    ) -> Result<Vec<crate::storage::ChunkWithEmbedding>> {
        let lance = self.lance.lock().await;
        let chunks = lance.get_all_chunks(source_id).await?;

        // Filter by source_id if specified
        let docs = self.documents.read().await;
        let chunk_to_doc = self.chunk_to_doc.read().await;

        let filtered: Vec<crate::storage::ChunkWithEmbedding> = chunks
            .into_iter()
            .filter(|(chunk_id, _, _, _)| {
                if let Some(sid) = source_id {
                    // Check if chunk's document belongs to the source
                    if let Some(doc_id) = chunk_to_doc.get(chunk_id) {
                        if let Some(doc) = docs.get(doc_id) {
                            return doc.source_id == sid;
                        }
                    }
                    false
                } else {
                    true
                }
            })
            .map(|(chunk_id, document_id, text, embedding)| {
                crate::storage::ChunkWithEmbedding {
                    chunk_id,
                    document_id,
                    text,
                    embedding,
                }
            })
            .collect();

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;

    fn create_test_doc(id: &str, content: &str) -> IndexedDocument {
        IndexedDocument {
            id: id.to_string(),
            source_id: "test-source".to_string(),
            path: format!("/test/{}.txt", id),
            mime_type: "text/plain".to_string(),
            size: content.len() as u64,
            content: content.to_string(),
            modified_at: Utc::now(),
            indexed_at: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    fn create_test_chunks(doc_id: &str, texts: &[&str], dimension: usize) -> Vec<VectorChunk> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                // Create a simple vector based on text hash
                let mut vector = vec![0.0f32; dimension];
                for (j, c) in text.chars().enumerate() {
                    vector[j % dimension] += (c as u32 as f32) / 1000.0;
                }
                // Normalize
                let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut vector {
                        *v /= norm;
                    }
                }

                VectorChunk {
                    id: format!("{}-chunk-{}", doc_id, i),
                    document_id: doc_id.to_string(),
                    text: text.to_string(),
                    vector,
                    start_offset: 0,
                    end_offset: text.len(),
                }
            })
            .collect()
    }

    #[tokio::test]
    async fn test_embedded_storage_create() {
        let temp_dir = TempDir::new().unwrap();
        let storage = EmbeddedStorage::new(temp_dir.path(), 384).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_embedded_store_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 64;
        let storage = EmbeddedStorage::new(temp_dir.path(), dimension)
            .await
            .unwrap();

        let doc = create_test_doc("doc1", "The quick brown fox jumps over the lazy dog");
        let chunks = create_test_chunks(
            "doc1",
            &["The quick brown fox", "jumps over the lazy dog"],
            dimension,
        );

        storage.store(doc.clone(), chunks).await.unwrap();

        let retrieved = storage.get("doc1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "doc1");
    }

    #[tokio::test]
    async fn test_embedded_text_search() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 64;
        let storage = EmbeddedStorage::new(temp_dir.path(), dimension)
            .await
            .unwrap();

        let doc = create_test_doc(
            "doc1",
            "Machine learning is a subset of artificial intelligence",
        );
        let chunks = create_test_chunks(
            "doc1",
            &["Machine learning is a subset", "of artificial intelligence"],
            dimension,
        );

        storage.store(doc, chunks).await.unwrap();

        // Text-only search
        let query = SearchQuery {
            text: "machine learning".to_string(),
            embedding: None,
            limit: 10,
            vector_weight: 0.0, // Full-text only
            source_id: None,
            file_types: vec![],
        };

        let results = storage.search(query).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].document_id, "doc1");
        assert!(results[0].text_score.is_some());
    }

    #[tokio::test]
    async fn test_embedded_hybrid_search() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 64;
        let storage = EmbeddedStorage::new(temp_dir.path(), dimension)
            .await
            .unwrap();

        // Index two documents
        let doc1 = create_test_doc("doc1", "Rust programming language is fast and safe");
        let chunks1 = create_test_chunks(
            "doc1",
            &["Rust programming language", "is fast and safe"],
            dimension,
        );

        let doc2 = create_test_doc(
            "doc2",
            "Python is great for data science and machine learning",
        );
        let chunks2 = create_test_chunks(
            "doc2",
            &["Python is great for data science", "and machine learning"],
            dimension,
        );

        storage.store(doc1, chunks1.clone()).await.unwrap();
        storage.store(doc2, chunks2).await.unwrap();

        // Hybrid search
        let query = SearchQuery {
            text: "Rust programming".to_string(),
            embedding: Some(chunks1[0].vector.clone()), // Use doc1's vector
            limit: 10,
            vector_weight: 0.5, // 50/50 hybrid
            source_id: None,
            file_types: vec![],
        };

        let results = storage.search(query).await.unwrap();
        assert!(!results.is_empty());
        // Doc1 should rank highest due to both text and vector match
        assert_eq!(results[0].document_id, "doc1");
    }

    #[tokio::test]
    async fn test_embedded_remove() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 64;
        let storage = EmbeddedStorage::new(temp_dir.path(), dimension)
            .await
            .unwrap();

        let doc = create_test_doc("doc1", "Test document for removal");
        let chunks = create_test_chunks("doc1", &["Test document", "for removal"], dimension);

        storage.store(doc, chunks).await.unwrap();

        // Verify it exists
        assert!(storage.get("doc1").await.unwrap().is_some());

        // Remove
        storage.remove("doc1").await.unwrap();

        // Verify it's gone
        assert!(storage.get("doc1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_embedded_stats() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 64;
        let storage = EmbeddedStorage::new(temp_dir.path(), dimension)
            .await
            .unwrap();

        let doc = create_test_doc("doc1", "Test document");
        let chunks = create_test_chunks("doc1", &["chunk1", "chunk2", "chunk3"], dimension);

        storage.store(doc, chunks).await.unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 3);
    }
}
