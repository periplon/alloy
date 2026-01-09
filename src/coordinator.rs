//! Index coordinator that orchestrates the full indexing pipeline.
//!
//! The coordinator handles:
//! - Source scanning and file watching
//! - Document processing and chunking
//! - Embedding generation
//! - Storage in both full-text and vector backends
//! - Progress reporting via callbacks

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::config::{Config, EmbeddingProvider as EmbeddingProviderType, StorageBackendType};
use crate::embedding::{
    ApiEmbeddingProvider, BatchConfig, BatchEmbeddingProcessor, EmbeddingProvider,
    LocalEmbeddingProvider,
};
use crate::error::Result;
use crate::processing::{
    CompositeDeduplicator, DeduplicationResult, Deduplicator, ProcessorRegistry, TextChunk,
};
use crate::search::{
    EnhancedHybridSearchBuilder, EnhancedHybridSearchOrchestrator, HybridQuery,
    HybridSearchResponse, HybridSearcher,
};
use crate::sources::{
    LocalSource, LocalSourceConfig, S3Source, S3SourceConfig, Source, SourceEvent, SourceItem,
};
use crate::storage::{
    EmbeddedStorage, IndexedDocument, QdrantHybridStorage, StorageBackend, StorageStats,
    VectorChunk,
};

/// Progress event during indexing.
#[derive(Debug, Clone)]
pub enum IndexProgress {
    /// Started scanning a source.
    ScanStarted { source_id: String },
    /// Finished scanning, found N items.
    ScanComplete {
        source_id: String,
        items_found: usize,
    },
    /// Started processing a document.
    ProcessingDocument {
        source_id: String,
        uri: String,
        current: usize,
        total: usize,
    },
    /// Finished processing a document.
    DocumentProcessed {
        source_id: String,
        uri: String,
        chunks: usize,
    },
    /// Document was skipped due to deduplication.
    DocumentDeduplicated {
        source_id: String,
        uri: String,
        duplicate_of: String,
        similarity: f32,
    },
    /// Error processing a document.
    DocumentError {
        source_id: String,
        uri: String,
        error: String,
    },
    /// Started embedding batch.
    EmbeddingBatch {
        source_id: String,
        batch_num: usize,
        total_batches: usize,
    },
    /// Finished storing documents.
    StorageComplete {
        source_id: String,
        documents: usize,
        chunks: usize,
    },
    /// Source indexing complete.
    IndexComplete {
        source_id: String,
        documents: usize,
        chunks: usize,
        duration_ms: u64,
    },
    /// File watch event.
    WatchEvent {
        source_id: String,
        event_type: String,
        uri: String,
    },
}

/// Result of indexing a single item.
#[derive(Debug)]
pub enum IndexItemResult {
    /// Document was indexed successfully with the given number of chunks.
    Indexed(usize),
    /// Document was skipped due to deduplication.
    Skipped(DeduplicationResult),
}

/// Indexed source information.
#[derive(Debug, Clone)]
pub struct IndexedSource {
    /// Source identifier.
    pub id: String,
    /// Source type ("local" or "s3").
    pub source_type: String,
    /// Source path or URI.
    pub path: String,
    /// Number of documents indexed.
    pub document_count: usize,
    /// Whether file watching is enabled.
    pub watching: bool,
    /// Last scan time.
    pub last_scan: chrono::DateTime<Utc>,
}

/// The main index coordinator.
pub struct IndexCoordinator {
    /// Configuration.
    #[allow(dead_code)]
    config: Config,
    /// Processor registry.
    processors: ProcessorRegistry,
    /// Embedding provider (with batching support).
    embedder: Arc<dyn EmbeddingProvider>,
    /// Storage backend.
    storage: Arc<dyn StorageBackend>,
    /// Indexed sources by ID.
    sources: RwLock<HashMap<String, IndexedSource>>,
    /// Active source watchers.
    #[allow(dead_code)]
    watchers: RwLock<HashMap<String, mpsc::Sender<()>>>,
    /// Progress callback sender.
    progress_tx: Option<mpsc::UnboundedSender<IndexProgress>>,
    /// Enhanced hybrid search orchestrator with reranking and query expansion.
    searcher: Arc<EnhancedHybridSearchOrchestrator>,
    /// Document deduplicator for detecting duplicate content.
    deduplicator: Option<Arc<dyn Deduplicator>>,
}

impl IndexCoordinator {
    /// Create a new index coordinator from configuration.
    pub async fn new(config: Config) -> Result<Self> {
        let data_dir = config.data_dir()?;
        std::fs::create_dir_all(&data_dir)?;

        // Create base embedding provider
        let base_embedder: Box<dyn EmbeddingProvider> = match config.embedding.provider {
            EmbeddingProviderType::Local => {
                Box::new(LocalEmbeddingProvider::new(&config.embedding.model, false)?)
            }
            EmbeddingProviderType::Api => {
                let api_config = &config.embedding.api;
                Box::new(ApiEmbeddingProvider::from_config(api_config)?)
            }
        };

        let dimension = base_embedder.dimension();

        // Wrap in batch processor with rate limiting
        let batch_config = BatchConfig::default()
            .with_batch_size(base_embedder.max_batch_size())
            .with_rate_limit(if config.embedding.provider == EmbeddingProviderType::Api {
                10
            } else {
                0
            });

        // We need to use a concrete type for BatchEmbeddingProcessor
        // For simplicity, we'll just use the base embedder wrapped in Arc
        let embedder: Arc<dyn EmbeddingProvider> = match config.embedding.provider {
            EmbeddingProviderType::Local => {
                let local = LocalEmbeddingProvider::new(&config.embedding.model, false)?;
                Arc::new(BatchEmbeddingProcessor::new(local, batch_config))
            }
            EmbeddingProviderType::Api => {
                let api = ApiEmbeddingProvider::from_config(&config.embedding.api)?;
                Arc::new(BatchEmbeddingProcessor::new(api, batch_config))
            }
        };

        // Create storage backend
        let storage: Arc<dyn StorageBackend> = match config.storage.backend {
            StorageBackendType::Embedded => {
                Arc::new(EmbeddedStorage::new(&data_dir, dimension).await?)
            }
            StorageBackendType::Qdrant => {
                let qdrant_config = &config.storage.qdrant;
                Arc::new(QdrantHybridStorage::new(qdrant_config, &data_dir, dimension).await?)
            }
        };

        // Create processor registry
        let processors = ProcessorRegistry::new(&config.processing);

        // Create enhanced hybrid searcher with reranking and query expansion support
        let searcher = EnhancedHybridSearchBuilder::new()
            .storage(storage.clone())
            .embedder(embedder.clone())
            .search_config(config.search.clone())
            .build()?;

        // Create deduplicator if enabled
        let deduplicator: Option<Arc<dyn Deduplicator>> =
            if config.indexing.deduplication.enabled {
                let dedup_config = crate::processing::DeduplicationConfig {
                    enabled: config.indexing.deduplication.enabled,
                    strategy: match config.indexing.deduplication.strategy {
                        crate::config::DeduplicationStrategy::Exact => {
                            crate::processing::DeduplicationStrategy::Exact
                        }
                        crate::config::DeduplicationStrategy::MinHash => {
                            crate::processing::DeduplicationStrategy::MinHash
                        }
                        crate::config::DeduplicationStrategy::Semantic => {
                            crate::processing::DeduplicationStrategy::Semantic
                        }
                    },
                    threshold: config.indexing.deduplication.threshold,
                    action: match config.indexing.deduplication.action {
                        crate::config::DeduplicationAction::Skip => {
                            crate::processing::DeduplicationAction::Skip
                        }
                        crate::config::DeduplicationAction::Flag => {
                            crate::processing::DeduplicationAction::Flag
                        }
                        crate::config::DeduplicationAction::Update => {
                            crate::processing::DeduplicationAction::Update
                        }
                    },
                    minhash_num_hashes: config.indexing.deduplication.minhash_num_hashes,
                    shingle_size: config.indexing.deduplication.shingle_size,
                };

                let dedup = CompositeDeduplicator::from_config(&dedup_config, Some(embedder.clone()));
                Some(Arc::new(dedup))
            } else {
                None
            };

        Ok(Self {
            config,
            processors,
            embedder,
            storage,
            sources: RwLock::new(HashMap::new()),
            watchers: RwLock::new(HashMap::new()),
            progress_tx: None,
            searcher: Arc::new(searcher),
            deduplicator,
        })
    }

    /// Set the progress callback channel.
    pub fn set_progress_channel(&mut self, tx: mpsc::UnboundedSender<IndexProgress>) {
        self.progress_tx = Some(tx);
    }

    /// Get a progress receiver.
    pub fn progress_channel(&mut self) -> mpsc::UnboundedReceiver<IndexProgress> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.progress_tx = Some(tx);
        rx
    }

    /// Report progress.
    fn report_progress(&self, progress: IndexProgress) {
        if let Some(tx) = &self.progress_tx {
            if let Err(e) = tx.send(progress.clone()) {
                debug!("Failed to send progress event: {}", e);
            }
        }
        // Also log the progress
        match &progress {
            IndexProgress::ScanStarted { source_id } => {
                info!("Scan started: {}", source_id);
            }
            IndexProgress::ScanComplete {
                source_id,
                items_found,
            } => {
                info!("Scan complete: {} ({} items)", source_id, items_found);
            }
            IndexProgress::IndexComplete {
                source_id,
                documents,
                chunks,
                duration_ms,
            } => {
                info!(
                    "Index complete: {} ({} documents, {} chunks, {}ms)",
                    source_id, documents, chunks, duration_ms
                );
            }
            IndexProgress::DocumentError { uri, error, .. } => {
                warn!("Document error {}: {}", uri, error);
            }
            _ => {
                debug!("Progress: {:?}", progress);
            }
        }
    }

    /// Index a local path.
    pub async fn index_local(
        &self,
        path: PathBuf,
        patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        watch: bool,
    ) -> Result<IndexedSource> {
        let config = LocalSourceConfig {
            path,
            patterns: if patterns.is_empty() {
                vec!["**/*".to_string()]
            } else {
                patterns
            },
            exclude_patterns: if exclude_patterns.is_empty() {
                LocalSourceConfig::default().exclude_patterns
            } else {
                exclude_patterns
            },
            watch,
            follow_symlinks: false,
        };

        let source = LocalSource::new(config)?;
        self.index_source(Box::new(source), "local", watch).await
    }

    /// Index an S3 path.
    pub async fn index_s3(
        &self,
        bucket: String,
        prefix: Option<String>,
        patterns: Vec<String>,
        region: Option<String>,
    ) -> Result<IndexedSource> {
        let config = S3SourceConfig {
            bucket,
            prefix,
            region,
            patterns: if patterns.is_empty() {
                vec!["*".to_string()]
            } else {
                patterns
            },
            ..Default::default()
        };

        let source = S3Source::new(config).await?;
        self.index_source(Box::new(source), "s3", false).await
    }

    /// Index a source.
    async fn index_source(
        &self,
        source: Box<dyn Source>,
        source_type: &str,
        _watch: bool,
    ) -> Result<IndexedSource> {
        let source_id = source.id().to_string();
        let start_time = std::time::Instant::now();

        self.report_progress(IndexProgress::ScanStarted {
            source_id: source_id.clone(),
        });

        // Scan the source
        let items = source.scan().await?;
        let total_items = items.len();

        self.report_progress(IndexProgress::ScanComplete {
            source_id: source_id.clone(),
            items_found: total_items,
        });

        // Process and index each document
        let mut documents_indexed = 0;
        let mut chunks_created = 0;

        for (idx, item) in items.iter().enumerate() {
            self.report_progress(IndexProgress::ProcessingDocument {
                source_id: source_id.clone(),
                uri: item.uri.clone(),
                current: idx + 1,
                total: total_items,
            });

            match self
                .process_and_index_item(&source_id, &*source, item)
                .await
            {
                Ok(IndexItemResult::Indexed(chunk_count)) => {
                    documents_indexed += 1;
                    chunks_created += chunk_count;
                    self.report_progress(IndexProgress::DocumentProcessed {
                        source_id: source_id.clone(),
                        uri: item.uri.clone(),
                        chunks: chunk_count,
                    });
                }
                Ok(IndexItemResult::Skipped(_)) => {
                    // Document was skipped due to deduplication
                    // Progress was already reported in process_and_index_item
                }
                Err(e) => {
                    self.report_progress(IndexProgress::DocumentError {
                        source_id: source_id.clone(),
                        uri: item.uri.clone(),
                        error: e.to_string(),
                    });
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        self.report_progress(IndexProgress::IndexComplete {
            source_id: source_id.clone(),
            documents: documents_indexed,
            chunks: chunks_created,
            duration_ms,
        });

        let indexed_source = IndexedSource {
            id: source_id.clone(),
            source_type: source_type.to_string(),
            path: source.id().to_string(),
            document_count: documents_indexed,
            watching: false, // TODO: implement watching
            last_scan: Utc::now(),
        };

        // Store in sources map
        {
            let mut sources = self.sources.write().await;
            sources.insert(source_id, indexed_source.clone());
        }

        Ok(indexed_source)
    }

    /// Process and index a single item.
    /// Returns Ok(Some(chunks)) if indexed, Ok(None) if skipped due to deduplication.
    async fn process_and_index_item(
        &self,
        source_id: &str,
        source: &dyn Source,
        item: &SourceItem,
    ) -> Result<IndexItemResult> {
        // Fetch content
        let content = source.fetch(&item.uri).await?;

        // Process the document
        let processed = self.processors.process(content, item).await?;

        // Check for duplicates if deduplication is enabled
        if let Some(dedup) = &self.deduplicator {
            let dedup_result = dedup.check(&processed.text, &item.id).await?;

            if dedup_result.is_duplicate {
                let action = self.config.indexing.deduplication.action;

                match action {
                    crate::config::DeduplicationAction::Skip => {
                        // Report duplicate and skip
                        self.report_progress(IndexProgress::DocumentDeduplicated {
                            source_id: source_id.to_string(),
                            uri: item.uri.clone(),
                            duplicate_of: dedup_result
                                .duplicate_of
                                .clone()
                                .unwrap_or_default(),
                            similarity: dedup_result.similarity,
                        });
                        return Ok(IndexItemResult::Skipped(dedup_result));
                    }
                    crate::config::DeduplicationAction::Flag => {
                        // Continue indexing but add duplicate flag to metadata
                        debug!(
                            "Document {} flagged as duplicate of {}",
                            item.id,
                            dedup_result.duplicate_of.as_deref().unwrap_or("unknown")
                        );
                    }
                    crate::config::DeduplicationAction::Update => {
                        // Continue indexing (will replace existing document)
                        debug!(
                            "Document {} will update existing duplicate",
                            item.id
                        );
                    }
                }
            }
        }

        // Generate embeddings for chunks
        let embeddings = self.embed_chunks(&processed.chunks).await?;

        // Build metadata
        let metadata = serde_json::json!({
            "title": processed.metadata.title,
            "author": processed.metadata.author,
            "word_count": processed.metadata.word_count,
        });

        // Create indexed document
        let doc = IndexedDocument {
            id: item.id.clone(),
            source_id: source_id.to_string(),
            path: item.uri.clone(),
            mime_type: item.mime_type.clone(),
            size: item.size,
            content: processed.text.clone(),
            modified_at: item.modified,
            indexed_at: Utc::now(),
            metadata,
        };

        // Create vector chunks
        let vector_chunks: Vec<VectorChunk> = processed
            .chunks
            .iter()
            .zip(embeddings.iter())
            .map(|(chunk, embedding)| VectorChunk {
                id: chunk.id.clone(),
                document_id: item.id.clone(),
                text: chunk.text.clone(),
                vector: embedding.clone(),
                start_offset: chunk.start_offset,
                end_offset: chunk.end_offset,
            })
            .collect();

        let chunk_count = vector_chunks.len();

        // Store in backend
        self.storage.store(doc, vector_chunks).await?;

        // Register with deduplicator for future checks
        if let Some(dedup) = &self.deduplicator {
            dedup.register(&processed.text, &item.id).await?;
        }

        Ok(IndexItemResult::Indexed(chunk_count))
    }

    /// Embed text chunks.
    async fn embed_chunks(&self, chunks: &[TextChunk]) -> Result<Vec<Vec<f32>>> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        self.embedder.embed(&texts).await
    }

    /// Search indexed documents.
    pub async fn search(&self, query: HybridQuery) -> Result<HybridSearchResponse> {
        self.searcher.search(query).await
    }

    /// Get a document by ID.
    pub async fn get_document(&self, doc_id: &str) -> Result<Option<IndexedDocument>> {
        self.storage.get(doc_id).await
    }

    /// Remove a source and all its documents.
    pub async fn remove_source(&self, source_id: &str) -> Result<usize> {
        // Get all documents for this source and remove them
        // For now, we'll just remove from our tracking
        // A full implementation would query storage for all docs with this source_id

        let mut sources = self.sources.write().await;
        if let Some(source) = sources.remove(source_id) {
            Ok(source.document_count)
        } else {
            Ok(0)
        }
    }

    /// List all indexed sources.
    pub async fn list_sources(&self) -> Vec<IndexedSource> {
        let sources = self.sources.read().await;
        sources.values().cloned().collect()
    }

    /// Get storage statistics.
    pub async fn stats(&self) -> Result<StorageStats> {
        self.storage.stats().await
    }

    /// Get the embedding dimension.
    pub fn embedding_dimension(&self) -> usize {
        self.embedder.dimension()
    }

    /// Handle a source event (for file watching).
    pub async fn handle_source_event(&self, source_id: &str, event: SourceEvent) -> Result<()> {
        match event {
            SourceEvent::Created(item) | SourceEvent::Modified(item) => {
                self.report_progress(IndexProgress::WatchEvent {
                    source_id: source_id.to_string(),
                    event_type: "modified".to_string(),
                    uri: item.uri.clone(),
                });

                // Re-index the item
                // We need the source to fetch content - this is a limitation
                // In a full implementation, we'd store source references
                info!("File changed: {} - would re-index", item.uri);
            }
            SourceEvent::Deleted(uri) => {
                self.report_progress(IndexProgress::WatchEvent {
                    source_id: source_id.to_string(),
                    event_type: "deleted".to_string(),
                    uri: uri.clone(),
                });

                // Remove from storage - would need to map URI to doc ID
                info!("File deleted: {} - would remove from index", uri);
            }
        }
        Ok(())
    }

    /// Cluster indexed documents by semantic similarity.
    ///
    /// Groups documents into clusters based on their embedding vectors.
    /// Returns clustering results including cluster labels, sizes, and quality metrics.
    pub async fn cluster_documents(
        &self,
        source_id: Option<&str>,
        algorithm: Option<crate::config::ClusteringAlgorithm>,
        num_clusters: Option<usize>,
    ) -> Result<crate::search::ClusteringResult> {
        use crate::search::{ClusterInput, ClusteringEngine};

        // Get all chunks with embeddings
        let chunks = self.storage.get_all_chunks_for_clustering(source_id).await?;

        if chunks.is_empty() {
            return Ok(crate::search::ClusteringResult {
                clusters: vec![],
                outliers: vec![],
                metrics: crate::search::ClusteringMetrics {
                    silhouette_score: 0.0,
                    inertia: 0.0,
                    num_clusters: 0,
                    num_outliers: 0,
                    cluster_size_distribution: vec![],
                },
                algorithm_used: "none".to_string(),
                total_documents: 0,
                created_at: chrono::Utc::now(),
            });
        }

        // Convert chunks to cluster inputs
        let inputs: Vec<ClusterInput> = chunks
            .into_iter()
            .map(|c| ClusterInput {
                document_id: c.document_id,
                embedding: c.embedding,
                text: Some(c.text),
            })
            .collect();

        // Build clustering config
        let mut cluster_config = self.config.search.clustering.clone();
        if let Some(algo) = algorithm {
            cluster_config.algorithm = algo;
        }
        if let Some(n) = num_clusters {
            cluster_config.default_num_clusters = n;
        }

        // Run clustering
        let engine = ClusteringEngine::new(self.embedder.clone(), cluster_config.clone());
        engine.cluster_documents(inputs, Some(&cluster_config)).await
    }

    // ========================================================================
    // Deduplication Methods
    // ========================================================================

    /// Check if content is a duplicate of an existing document.
    ///
    /// Returns the deduplication result with similarity information.
    pub async fn check_duplicate(
        &self,
        content: &str,
        doc_id: Option<&str>,
    ) -> Result<Option<DeduplicationResult>> {
        if let Some(dedup) = &self.deduplicator {
            let id = doc_id.unwrap_or("check");
            let result = dedup.check(content, id).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Check if deduplication is enabled.
    pub fn is_deduplication_enabled(&self) -> bool {
        self.deduplicator.is_some()
    }

    /// Get deduplication configuration.
    pub fn deduplication_config(&self) -> &crate::config::DeduplicationConfig {
        &self.config.indexing.deduplication
    }

    /// Clear the deduplication index.
    ///
    /// This removes all registered documents from the deduplicator.
    pub async fn clear_deduplication_index(&self) -> Result<()> {
        if let Some(dedup) = &self.deduplicator {
            dedup.clear().await?;
        }
        Ok(())
    }

    /// Register content with the deduplicator for future duplicate checking.
    pub async fn register_for_deduplication(&self, content: &str, doc_id: &str) -> Result<()> {
        if let Some(dedup) = &self.deduplicator {
            dedup.register(content, doc_id).await?;
        }
        Ok(())
    }

    /// Remove a document from the deduplication index.
    pub async fn remove_from_deduplication(&self, doc_id: &str) -> Result<()> {
        if let Some(dedup) = &self.deduplicator {
            dedup.remove(doc_id).await?;
        }
        Ok(())
    }
}

/// Builder for IndexCoordinator.
pub struct IndexCoordinatorBuilder {
    config: Option<Config>,
    progress_tx: Option<mpsc::UnboundedSender<IndexProgress>>,
}

impl Default for IndexCoordinatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexCoordinatorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            progress_tx: None,
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Set progress channel.
    pub fn progress_channel(mut self, tx: mpsc::UnboundedSender<IndexProgress>) -> Self {
        self.progress_tx = Some(tx);
        self
    }

    /// Build the coordinator.
    pub async fn build(self) -> Result<IndexCoordinator> {
        let config = self.config.unwrap_or_default();
        let mut coordinator = IndexCoordinator::new(config).await?;

        if let Some(tx) = self.progress_tx {
            coordinator.set_progress_channel(tx);
        }

        Ok(coordinator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_config(data_dir: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.storage.data_dir = data_dir.to_string_lossy().to_string();
        config
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_coordinator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(temp_dir.path());

        let coordinator = IndexCoordinator::new(config).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_index_local_directory() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = TempDir::new().unwrap();

        // Create test files
        let file1 = temp_dir.path().join("test1.txt");
        let mut f = File::create(&file1).unwrap();
        writeln!(f, "This is a test document about machine learning.").unwrap();

        let file2 = temp_dir.path().join("test2.txt");
        let mut f = File::create(&file2).unwrap();
        writeln!(f, "This is another document about artificial intelligence.").unwrap();

        let config = create_test_config(data_dir.path());
        let coordinator = IndexCoordinator::new(config).await.unwrap();

        let result = coordinator
            .index_local(
                temp_dir.path().to_path_buf(),
                vec!["*.txt".to_string()],
                vec![],
                false,
            )
            .await;

        assert!(result.is_ok());
        let source = result.unwrap();
        assert_eq!(source.document_count, 2);
    }

    #[tokio::test]
    #[ignore = "requires embedding model download"]
    async fn test_list_sources() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = TempDir::new().unwrap();

        // Create a test file
        let file1 = temp_dir.path().join("test.txt");
        let mut f = File::create(&file1).unwrap();
        writeln!(f, "Test content").unwrap();

        let config = create_test_config(data_dir.path());
        let coordinator = IndexCoordinator::new(config).await.unwrap();

        // Index the directory
        coordinator
            .index_local(temp_dir.path().to_path_buf(), vec![], vec![], false)
            .await
            .unwrap();

        let sources = coordinator.list_sources().await;
        assert_eq!(sources.len(), 1);
    }
}
