//! Qdrant vector database backend.
//!
//! This backend stores vectors in an external Qdrant instance and
//! uses Qdrant's built-in full-text search capabilities for hybrid search.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder, VectorsConfig,
};
use qdrant_client::Qdrant;

use crate::config::QdrantConfig;
use crate::error::{Result, StorageError};
use crate::search::rrf_fusion;
use crate::storage::{
    IndexedDocument, SearchQuery, StorageBackend, StorageSearchResult, StorageStats, VectorChunk,
};

/// Qdrant storage backend.
///
/// Provides unified vector and document storage with hybrid search capabilities.
pub struct QdrantStorage {
    client: Qdrant,
    collection_name: String,
    dimension: usize,
    documents: Arc<RwLock<HashMap<String, IndexedDocument>>>,
}

impl QdrantStorage {
    /// Create a new Qdrant storage backend.
    pub async fn new(config: &QdrantConfig, dimension: usize) -> Result<Self> {
        let client = Qdrant::from_url(&config.url)
            .api_key(config.api_key.clone())
            .build()
            .map_err(|e| StorageError::Connection(e.to_string()))?;

        let storage = Self {
            client,
            collection_name: config.collection.clone(),
            dimension,
            documents: Arc::new(RwLock::new(HashMap::new())),
        };

        // Ensure collection exists
        storage.ensure_collection().await?;

        Ok(storage)
    }

    /// Ensure the collection exists with correct configuration.
    async fn ensure_collection(&self) -> Result<()> {
        let collections = self
            .client
            .list_collections()
            .await
            .map_err(|e| StorageError::Connection(e.to_string()))?;

        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection_name);

        if !exists {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.collection_name).vectors_config(
                        VectorsConfig {
                            config: Some(Config::Params(
                                VectorParamsBuilder::new(self.dimension as u64, Distance::Cosine)
                                    .build(),
                            )),
                        },
                    ),
                )
                .await
                .map_err(|e| StorageError::Index(e.to_string()))?;

            tracing::info!("Created Qdrant collection: {}", self.collection_name);
        }

        Ok(())
    }

    /// Convert a chunk ID to a Qdrant point ID.
    fn chunk_id_to_point_id(chunk_id: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        chunk_id.hash(&mut hasher);
        hasher.finish()
    }
}

#[async_trait]
impl StorageBackend for QdrantStorage {
    async fn store(&self, doc: IndexedDocument, vectors: Vec<VectorChunk>) -> Result<()> {
        // Store document in memory
        {
            let mut docs = self.documents.write();
            docs.insert(doc.id.clone(), doc.clone());
        }

        // Convert chunks to Qdrant points
        let points: Vec<PointStruct> = vectors
            .iter()
            .map(|chunk| {
                PointStruct::new(
                    Self::chunk_id_to_point_id(&chunk.id),
                    chunk.vector.clone(),
                    [
                        ("chunk_id", chunk.id.as_str().into()),
                        ("document_id", chunk.document_id.as_str().into()),
                        ("text", chunk.text.as_str().into()),
                        ("source_id", doc.source_id.as_str().into()),
                        ("path", doc.path.as_str().into()),
                        ("mime_type", doc.mime_type.as_str().into()),
                        ("start_offset", (chunk.start_offset as i64).into()),
                        ("end_offset", (chunk.end_offset as i64).into()),
                    ],
                )
            })
            .collect();

        if !points.is_empty() {
            self.client
                .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
                .await
                .map_err(|e| StorageError::Index(e.to_string()))?;
        }

        tracing::debug!("Stored {} chunks for document {}", vectors.len(), doc.id);

        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<StorageSearchResult>> {
        // Vector search requires embedding
        let embedding = match &query.embedding {
            Some(e) => e,
            None => {
                return Err(
                    StorageError::Query("Qdrant requires embedding for search".to_string()).into(),
                )
            }
        };

        let limit = query.limit;

        // Build filter if source_id specified
        let filter = query.source_id.as_ref().map(|source_id| {
            Filter::must([qdrant_client::qdrant::Condition::matches(
                "source_id",
                source_id.clone(),
            )])
        });

        // Execute search
        let mut search_builder =
            SearchPointsBuilder::new(&self.collection_name, embedding.clone(), limit as u64)
                .with_payload(true);

        if let Some(f) = filter {
            search_builder = search_builder.filter(f);
        }

        let search_result = self
            .client
            .search_points(search_builder)
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        // Convert results
        let results: Vec<StorageSearchResult> = search_result
            .result
            .iter()
            .filter_map(|point| {
                let payload = &point.payload;

                let chunk_id = payload.get("chunk_id")?.as_str().map(|s| s.to_string())?;
                let document_id = payload
                    .get("document_id")?
                    .as_str()
                    .map(|s| s.to_string())?;
                let text = payload.get("text")?.as_str().map(|s| s.to_string())?;

                Some(StorageSearchResult {
                    document_id,
                    chunk_id,
                    text,
                    score: point.score,
                    vector_score: Some(point.score),
                    text_score: None, // Qdrant is vector-only in this implementation
                })
            })
            .collect();

        Ok(results)
    }

    async fn remove(&self, doc_id: &str) -> Result<()> {
        // Remove from memory
        {
            let mut docs = self.documents.write();
            docs.remove(doc_id);
        }

        // Delete points with matching document_id
        let filter = Filter::must([qdrant_client::qdrant::Condition::matches(
            "document_id",
            doc_id.to_string(),
        )]);

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(filter)
                    .wait(true),
            )
            .await
            .map_err(|e| StorageError::Index(e.to_string()))?;

        tracing::debug!("Removed document {}", doc_id);

        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Option<IndexedDocument>> {
        let docs = self.documents.read();
        Ok(docs.get(doc_id).cloned())
    }

    async fn stats(&self) -> Result<StorageStats> {
        let collection_info = self
            .client
            .collection_info(&self.collection_name)
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let info = collection_info
            .result
            .ok_or_else(|| StorageError::Query("Failed to get collection info".to_string()))?;

        let chunk_count = info.points_count.unwrap_or(0) as usize;
        let document_count = self.documents.read().len();

        Ok(StorageStats {
            document_count,
            chunk_count,
            storage_bytes: 0, // Qdrant doesn't expose this directly
        })
    }

    async fn get_all_chunks_for_clustering(
        &self,
        source_id: Option<&str>,
    ) -> Result<Vec<crate::storage::ChunkWithEmbedding>> {
        use qdrant_client::qdrant::ScrollPointsBuilder;

        // Build filter if source_id specified
        let filter = source_id.map(|sid| {
            Filter::must([qdrant_client::qdrant::Condition::matches(
                "source_id",
                sid.to_string(),
            )])
        });

        let mut all_chunks = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        // Scroll through all points
        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&self.collection_name)
                .limit(1000)
                .with_payload(true)
                .with_vectors(true);

            if let Some(f) = filter.clone() {
                scroll_builder = scroll_builder.filter(f);
            }

            if let Some(off) = offset.clone() {
                scroll_builder = scroll_builder.offset(off);
            }

            let scroll_result = self
                .client
                .scroll(scroll_builder)
                .await
                .map_err(|e| StorageError::Query(e.to_string()))?;

            let result = scroll_result.result;

            for point in &result {
                let payload = &point.payload;

                let chunk_id = payload
                    .get("chunk_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let document_id = payload
                    .get("document_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let text = payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();

                // Extract vector from point using the helper method
                let embedding = point
                    .vectors
                    .as_ref()
                    .and_then(|v| v.get_vector())
                    .and_then(|vec| {
                        use qdrant_client::qdrant::vector_output::Vector;
                        match vec {
                            Vector::Dense(dense) => Some(dense.data),
                            _ => None,
                        }
                    })
                    .unwrap_or_default();

                if !embedding.is_empty() {
                    all_chunks.push(crate::storage::ChunkWithEmbedding {
                        chunk_id,
                        document_id,
                        text,
                        embedding,
                    });
                }
            }

            // Check if we have more results
            if let Some(next_offset) = scroll_result.next_page_offset {
                offset = Some(next_offset);
            } else {
                break;
            }
        }

        Ok(all_chunks)
    }

    async fn get_all_documents(&self, source_id: Option<&str>) -> Result<Vec<IndexedDocument>> {
        let docs = self.documents.read();

        let filtered: Vec<IndexedDocument> = docs
            .values()
            .filter(|doc| {
                if let Some(sid) = source_id {
                    doc.source_id == sid
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        Ok(filtered)
    }

    async fn get_document_chunks(&self, doc_id: &str) -> Result<Vec<VectorChunk>> {
        use qdrant_client::qdrant::ScrollPointsBuilder;

        // Filter by document_id
        let filter = Filter::must([qdrant_client::qdrant::Condition::matches(
            "document_id",
            doc_id.to_string(),
        )]);

        let mut chunks = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;

        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&self.collection_name)
                .limit(1000)
                .filter(filter.clone())
                .with_payload(true)
                .with_vectors(true);

            if let Some(off) = offset.clone() {
                scroll_builder = scroll_builder.offset(off);
            }

            let scroll_result = self
                .client
                .scroll(scroll_builder)
                .await
                .map_err(|e| StorageError::Query(e.to_string()))?;

            let result = scroll_result.result;

            for point in &result {
                let payload = &point.payload;

                let chunk_id = payload
                    .get("chunk_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let text = payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let start_offset = payload
                    .get("start_offset")
                    .and_then(|v| v.as_integer())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                let end_offset = payload
                    .get("end_offset")
                    .and_then(|v| v.as_integer())
                    .map(|i| i as usize)
                    .unwrap_or(0);

                let vector = point
                    .vectors
                    .as_ref()
                    .and_then(|v| v.get_vector())
                    .and_then(|vec| {
                        use qdrant_client::qdrant::vector_output::Vector;
                        match vec {
                            Vector::Dense(dense) => Some(dense.data.clone()),
                            _ => None,
                        }
                    })
                    .unwrap_or_default();

                chunks.push(VectorChunk {
                    id: chunk_id,
                    document_id: doc_id.to_string(),
                    text,
                    vector,
                    start_offset,
                    end_offset,
                });
            }

            if let Some(next_offset) = scroll_result.next_page_offset {
                offset = Some(next_offset);
            } else {
                break;
            }
        }

        Ok(chunks)
    }
}

/// Qdrant hybrid storage with full-text support.
///
/// This variant uses Qdrant for vectors and a separate Tantivy instance for full-text.
pub struct QdrantHybridStorage {
    qdrant: QdrantStorage,
    tantivy: crate::storage::tantivy_backend::TantivyBackend,
}

impl QdrantHybridStorage {
    /// Create a new Qdrant hybrid storage.
    pub async fn new(
        config: &QdrantConfig,
        data_dir: &std::path::Path,
        dimension: usize,
    ) -> Result<Self> {
        let qdrant = QdrantStorage::new(config, dimension).await?;
        let tantivy = crate::storage::tantivy_backend::TantivyBackend::new(data_dir)?;

        Ok(Self { qdrant, tantivy })
    }
}

#[async_trait]
impl StorageBackend for QdrantHybridStorage {
    async fn store(&self, doc: IndexedDocument, vectors: Vec<VectorChunk>) -> Result<()> {
        // Store in both backends
        self.tantivy.index_document(&doc)?;
        self.tantivy.commit()?;

        self.qdrant.store(doc, vectors).await
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<StorageSearchResult>> {
        let limit = query.limit;
        let vector_weight = query.vector_weight.clamp(0.0, 1.0);
        let text_weight = 1.0 - vector_weight;

        let mut result_lists: Vec<Vec<(String, f32)>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();
        let mut text_map: HashMap<String, (String, f32)> = HashMap::new();
        let mut vector_map: HashMap<String, (String, f32)> = HashMap::new();

        // Full-text search
        if text_weight > 0.0 {
            let text_results =
                self.tantivy
                    .search(&query.text, query.source_id.as_deref(), limit * 2)?;

            for result in text_results {
                text_map.insert(
                    result.document_id.clone(),
                    (result.snippet.clone(), result.score),
                );
            }

            let text_list: Vec<_> = text_map
                .iter()
                .map(|(id, (_, score))| (id.clone(), *score))
                .collect();

            if !text_list.is_empty() {
                result_lists.push(text_list);
                weights.push(text_weight);
            }
        }

        // Vector search
        if query.embedding.is_some() && vector_weight > 0.0 {
            let vector_results = self.qdrant.search(query.clone()).await?;

            for result in vector_results {
                let entry = vector_map
                    .entry(result.document_id.clone())
                    .or_insert((result.text.clone(), 0.0));
                if result.score > entry.1 {
                    *entry = (result.text.clone(), result.score);
                }
            }

            let vector_list: Vec<_> = vector_map
                .iter()
                .map(|(id, (_, score))| (id.clone(), *score))
                .collect();

            if !vector_list.is_empty() {
                result_lists.push(vector_list);
                weights.push(vector_weight);
            }
        }

        // Fuse results
        let fused = rrf_fusion(&result_lists, 60.0, &weights);

        // Build final results
        let results: Vec<StorageSearchResult> = fused
            .into_iter()
            .take(limit)
            .map(|(doc_id, score)| {
                let text = vector_map
                    .get(&doc_id)
                    .map(|(t, _)| t.clone())
                    .or_else(|| text_map.get(&doc_id).map(|(t, _)| t.clone()))
                    .unwrap_or_default();

                StorageSearchResult {
                    document_id: doc_id.clone(),
                    chunk_id: doc_id.clone(),
                    text,
                    score,
                    vector_score: vector_map.get(&doc_id).map(|(_, s)| *s),
                    text_score: text_map.get(&doc_id).map(|(_, s)| *s),
                }
            })
            .collect();

        Ok(results)
    }

    async fn remove(&self, doc_id: &str) -> Result<()> {
        self.tantivy.remove(doc_id)?;
        self.tantivy.commit()?;
        self.qdrant.remove(doc_id).await
    }

    async fn get(&self, doc_id: &str) -> Result<Option<IndexedDocument>> {
        self.qdrant.get(doc_id).await
    }

    async fn stats(&self) -> Result<StorageStats> {
        let mut stats = self.qdrant.stats().await?;
        stats.document_count = self.tantivy.document_count()?;
        Ok(stats)
    }

    async fn get_all_chunks_for_clustering(
        &self,
        source_id: Option<&str>,
    ) -> Result<Vec<crate::storage::ChunkWithEmbedding>> {
        // Delegate to the Qdrant storage
        self.qdrant.get_all_chunks_for_clustering(source_id).await
    }

    async fn get_all_documents(&self, source_id: Option<&str>) -> Result<Vec<IndexedDocument>> {
        self.qdrant.get_all_documents(source_id).await
    }

    async fn get_document_chunks(&self, doc_id: &str) -> Result<Vec<VectorChunk>> {
        self.qdrant.get_document_chunks(doc_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running Qdrant instance
    // They are marked as ignore by default

    #[tokio::test]
    #[ignore]
    async fn test_qdrant_storage_create() {
        let config = QdrantConfig::default();
        let storage = QdrantStorage::new(&config, 384).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_qdrant_store_and_search() {
        let config = QdrantConfig {
            collection: "test_alloy".to_string(),
            ..Default::default()
        };
        let storage = QdrantStorage::new(&config, 8).await.unwrap();

        // Store a document
        let doc = IndexedDocument {
            id: "test-doc-1".to_string(),
            source_id: "test-source".to_string(),
            path: "/test/doc.txt".to_string(),
            mime_type: "text/plain".to_string(),
            size: 100,
            content: "Test content".to_string(),
            modified_at: chrono::Utc::now(),
            indexed_at: chrono::Utc::now(),
            metadata: serde_json::json!({}),
        };

        let chunks = vec![VectorChunk {
            id: "chunk-1".to_string(),
            document_id: "test-doc-1".to_string(),
            text: "Test content".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            start_offset: 0,
            end_offset: 12,
        }];

        storage.store(doc, chunks).await.unwrap();

        // Search
        let query = SearchQuery {
            text: "test".to_string(),
            embedding: Some(vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            limit: 10,
            vector_weight: 1.0,
            source_id: None,
            file_types: vec![],
            acl_user: None,
            acl_roles: None,
        };

        let results = storage.search(query).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].document_id, "test-doc-1");
    }
}
