//! LanceDB vector storage backend.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use lancedb::Connection;
use tokio::sync::RwLock;

use crate::error::{Result, StorageError};
use crate::storage::VectorChunk;

/// Result from a vector search.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Chunk ID
    pub chunk_id: String,
    /// Document ID
    pub document_id: String,
    /// Chunk text
    pub text: String,
    /// Similarity score (higher is better)
    pub score: f32,
}

/// LanceDB vector storage backend.
pub struct LanceBackend {
    connection: Connection,
    table: RwLock<Option<Table>>,
    dimension: usize,
    table_name: String,
}

impl LanceBackend {
    /// Create a new LanceDB backend.
    pub async fn new(data_dir: &Path, dimension: usize) -> Result<Self> {
        let db_path = data_dir.join("lancedb");
        std::fs::create_dir_all(&db_path).map_err(StorageError::Io)?;

        let connection = connect(db_path.to_string_lossy().as_ref())
            .execute()
            .await
            .map_err(|e| StorageError::Connection(e.to_string()))?;

        let table_name = "vectors".to_string();

        // Try to open existing table (table might not exist yet)
        let table = connection.open_table(&table_name).execute().await.ok();

        Ok(Self {
            connection,
            table: RwLock::new(table),
            dimension,
            table_name,
        })
    }

    /// Get the Arrow schema for the vectors table.
    fn schema(&self) -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("document_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimension as i32,
                ),
                false,
            ),
        ])
    }

    /// Create a record batch from vector chunks.
    fn create_batch(&self, chunks: &[VectorChunk]) -> Result<RecordBatch> {
        let chunk_ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let doc_ids: Vec<&str> = chunks.iter().map(|c| c.document_id.as_str()).collect();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

        // Flatten vectors for FixedSizeList
        let vectors: Vec<f32> = chunks.iter().flat_map(|c| c.vector.clone()).collect();

        let chunk_id_array = StringArray::from(chunk_ids);
        let doc_id_array = StringArray::from(doc_ids);
        let text_array = StringArray::from(texts);

        // Create FixedSizeListArray properly
        let values = Float32Array::from(vectors);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = arrow_array::FixedSizeListArray::new(
            field,
            self.dimension as i32,
            Arc::new(values),
            None,
        );

        Ok(RecordBatch::try_new(
            Arc::new(self.schema()),
            vec![
                Arc::new(chunk_id_array),
                Arc::new(doc_id_array),
                Arc::new(text_array),
                Arc::new(vector_array),
            ],
        )
        .map_err(|e| StorageError::Index(e.to_string()))?)
    }

    /// Ensure the table exists.
    async fn ensure_table(&self, batch: RecordBatch) -> Result<Table> {
        let mut table_guard = self.table.write().await;

        if table_guard.is_none() {
            // Create table with initial batch
            let schema = Arc::new(self.schema());
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

            let table = self
                .connection
                .create_table(&self.table_name, Box::new(batches))
                .execute()
                .await
                .map_err(|e| StorageError::Index(e.to_string()))?;

            *table_guard = Some(table.clone());
            Ok(table)
        } else {
            Ok(table_guard.clone().unwrap())
        }
    }

    /// Store vector chunks.
    pub async fn store(&self, chunks: Vec<VectorChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for chunk in &chunks {
            if chunk.vector.len() != self.dimension {
                return Err(StorageError::SchemaMismatch(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    self.dimension,
                    chunk.vector.len()
                ))
                .into());
            }
        }

        let batch = self.create_batch(&chunks)?;

        // Check if table exists
        let table_exists = {
            let guard = self.table.read().await;
            guard.is_some()
        };

        if !table_exists {
            // Ensure table exists (this will create it)
            self.ensure_table(batch).await?;
        } else {
            // Table exists, add data
            let table = {
                let guard = self.table.read().await;
                guard.clone().unwrap()
            };

            let schema = Arc::new(self.schema());
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

            table
                .add(Box::new(batches))
                .execute()
                .await
                .map_err(|e| StorageError::Index(e.to_string()))?;
        }

        Ok(())
    }

    /// Search for similar vectors.
    pub async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        _document_filter: Option<&str>,
    ) -> Result<Vec<VectorSearchResult>> {
        if query_vector.len() != self.dimension {
            return Err(StorageError::SchemaMismatch(format!(
                "Query vector dimension mismatch: expected {}, got {}",
                self.dimension,
                query_vector.len()
            ))
            .into());
        }

        let table = {
            let guard = self.table.read().await;
            match guard.as_ref() {
                Some(t) => t.clone(),
                None => return Ok(Vec::new()), // No data yet
            }
        };

        // Build and execute query
        let query = table
            .vector_search(query_vector.to_vec())
            .map_err(|e| StorageError::Query(e.to_string()))?
            .limit(limit);

        // Note: LanceDB filter syntax would go here if needed
        // if let Some(doc_id) = document_filter {
        //     query = query.filter(format!("document_id = '{}'", doc_id));
        // }

        // Execute query
        let results = query
            .execute()
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        // Collect results
        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut search_results = Vec::new();

        for batch in batches {
            let chunk_ids = batch
                .column_by_name("chunk_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let doc_ids = batch
                .column_by_name("document_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let texts = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(chunk_ids), Some(doc_ids), Some(texts), Some(distances)) =
                (chunk_ids, doc_ids, texts, distances)
            {
                for i in 0..batch.num_rows() {
                    let chunk_id = chunk_ids.value(i).to_string();
                    let document_id = doc_ids.value(i).to_string();
                    let text = texts.value(i).to_string();
                    let distance = distances.value(i);

                    // Convert L2 distance to similarity score (1 / (1 + distance))
                    let score = 1.0 / (1.0 + distance);

                    search_results.push(VectorSearchResult {
                        chunk_id,
                        document_id,
                        text,
                        score,
                    });
                }
            }
        }

        Ok(search_results)
    }

    /// Remove all chunks for a document.
    pub async fn remove_document(&self, doc_id: &str) -> Result<()> {
        let table = {
            let guard = self.table.read().await;
            match guard.as_ref() {
                Some(t) => t.clone(),
                None => return Ok(()),
            }
        };

        table
            .delete(&format!("document_id = '{}'", doc_id))
            .await
            .map_err(|e| StorageError::Index(e.to_string()))?;

        Ok(())
    }

    /// Get the number of chunks.
    pub async fn chunk_count(&self) -> Result<usize> {
        let table = {
            let guard = self.table.read().await;
            match guard.as_ref() {
                Some(t) => t.clone(),
                None => return Ok(0),
            }
        };

        let count = table
            .count_rows(None)
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        Ok(count)
    }

    /// Get all chunks with their embeddings for clustering.
    ///
    /// Returns a vector of (chunk_id, document_id, text, embedding) tuples.
    pub async fn get_all_chunks(
        &self,
        source_filter: Option<&str>,
    ) -> Result<Vec<(String, String, String, Vec<f32>)>> {
        let table = {
            let guard = self.table.read().await;
            match guard.as_ref() {
                Some(t) => t.clone(),
                None => return Ok(Vec::new()),
            }
        };

        // Build query to get all chunks
        let query = table.query();

        // Note: source filtering would need to be added to the schema if needed

        let results = query
            .execute()
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut chunks = Vec::new();

        for batch in batches {
            let chunk_ids = batch
                .column_by_name("chunk_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let doc_ids = batch
                .column_by_name("document_id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let texts = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let vectors = batch
                .column_by_name("vector")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::FixedSizeListArray>());

            if let (Some(chunk_ids), Some(doc_ids), Some(texts), Some(vectors)) =
                (chunk_ids, doc_ids, texts, vectors)
            {
                for i in 0..batch.num_rows() {
                    let chunk_id = chunk_ids.value(i).to_string();
                    let document_id = doc_ids.value(i).to_string();
                    let text = texts.value(i).to_string();

                    // Extract the vector from FixedSizeListArray
                    let vector_arr = vectors.value(i);
                    let float_arr = vector_arr.as_any().downcast_ref::<Float32Array>().unwrap();
                    let embedding: Vec<f32> =
                        (0..float_arr.len()).map(|j| float_arr.value(j)).collect();

                    // Apply source filter if specified
                    if let Some(_filter) = source_filter {
                        // TODO: source filtering requires source_id in the schema
                    }

                    chunks.push((chunk_id, document_id, text, embedding));
                }
            }
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_chunk(id: &str, doc_id: &str, text: &str, vector: Vec<f32>) -> VectorChunk {
        VectorChunk {
            id: id.to_string(),
            document_id: doc_id.to_string(),
            text: text.to_string(),
            vector,
            start_offset: 0,
            end_offset: text.len(),
        }
    }

    #[tokio::test]
    async fn test_lance_backend_create() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceBackend::new(temp_dir.path(), 384).await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_lance_store_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 8; // Small dimension for testing
        let backend = LanceBackend::new(temp_dir.path(), dimension).await.unwrap();

        // Create test chunks with different vectors
        let chunks = vec![
            create_test_chunk(
                "chunk1",
                "doc1",
                "Hello world",
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            create_test_chunk(
                "chunk2",
                "doc1",
                "Goodbye world",
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            create_test_chunk(
                "chunk3",
                "doc2",
                "Different document",
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ];

        backend.store(chunks).await.unwrap();

        // Search for similar to first chunk
        let query_vector = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = backend.search(&query_vector, 10, None).await.unwrap();

        assert!(!results.is_empty());
        // First result should be closest to query
        assert_eq!(results[0].chunk_id, "chunk1");
    }

    #[tokio::test]
    async fn test_lance_dimension_validation() {
        let temp_dir = TempDir::new().unwrap();
        let backend = LanceBackend::new(temp_dir.path(), 8).await.unwrap();

        // Try to store chunk with wrong dimension
        let chunk = create_test_chunk(
            "chunk1",
            "doc1",
            "Test",
            vec![1.0, 0.0, 0.0], // Wrong dimension
        );

        let result = backend.store(vec![chunk]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_lance_remove_document() {
        let temp_dir = TempDir::new().unwrap();
        let dimension = 8;
        let backend = LanceBackend::new(temp_dir.path(), dimension).await.unwrap();

        // Store chunks
        let chunks = vec![
            create_test_chunk(
                "chunk1",
                "doc1",
                "First doc",
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            create_test_chunk(
                "chunk2",
                "doc2",
                "Second doc",
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ];

        backend.store(chunks).await.unwrap();

        // Remove doc1
        backend.remove_document("doc1").await.unwrap();

        // Search should only return doc2
        let query = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = backend.search(&query, 10, None).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc2");
    }
}
