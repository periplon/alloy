//! Backup and restore functionality for Alloy MCP server.
//!
//! This module provides tools for:
//! - Creating full index backups (Tantivy + LanceDB data)
//! - Restoring from backups
//! - Exporting documents to portable formats (JSONL, etc.)
//! - Importing documents from exports

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::storage::{IndexedDocument, StorageBackend, VectorChunk};

/// Backup metadata stored alongside the backup data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup identifier.
    pub backup_id: String,
    /// When the backup was created.
    pub created_at: DateTime<Utc>,
    /// Alloy version that created the backup.
    pub version: String,
    /// Number of documents in the backup.
    pub document_count: usize,
    /// Number of chunks in the backup.
    pub chunk_count: usize,
    /// Total size in bytes.
    pub size_bytes: u64,
    /// Storage backend type.
    pub storage_backend: String,
    /// Embedding dimension.
    pub embedding_dimension: usize,
    /// Optional description.
    pub description: Option<String>,
}

impl BackupMetadata {
    /// Create new backup metadata.
    pub fn new(
        document_count: usize,
        chunk_count: usize,
        storage_backend: &str,
        embedding_dimension: usize,
    ) -> Self {
        Self {
            backup_id: uuid::Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            document_count,
            chunk_count,
            size_bytes: 0,
            storage_backend: storage_backend.to_string(),
            embedding_dimension,
            description: None,
        }
    }

    /// Add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Result of a backup operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupResult {
    /// Backup metadata.
    pub metadata: BackupMetadata,
    /// Path to the backup file.
    pub path: PathBuf,
    /// Duration of the backup operation.
    pub duration_ms: u64,
}

/// Result of a restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    /// Original backup metadata.
    pub metadata: BackupMetadata,
    /// Number of documents restored.
    pub documents_restored: usize,
    /// Number of chunks restored.
    pub chunks_restored: usize,
    /// Duration of the restore operation.
    pub duration_ms: u64,
}

/// Export format for documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// JSON Lines format (one JSON object per line).
    #[default]
    Jsonl,
    /// Pretty-printed JSON array.
    Json,
}

/// Export options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format.
    pub format: ExportFormat,
    /// Include document content.
    pub include_content: bool,
    /// Include embeddings.
    pub include_embeddings: bool,
    /// Filter by source ID.
    pub source_id: Option<String>,
    /// Compress output.
    pub compress: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Jsonl,
            include_content: true,
            include_embeddings: false, // Embeddings are large
            source_id: None,
            compress: false,
        }
    }
}

/// Result of an export operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Path to the export file.
    pub path: PathBuf,
    /// Number of documents exported.
    pub document_count: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Duration of the export operation.
    pub duration_ms: u64,
}

/// Document export record (for JSONL/JSON export).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentExport {
    /// Document ID.
    pub id: String,
    /// Source ID.
    pub source_id: String,
    /// Document path.
    pub path: String,
    /// MIME type.
    pub mime_type: String,
    /// Size in bytes.
    pub size: u64,
    /// Document content (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Modification time.
    pub modified_at: DateTime<Utc>,
    /// Indexing time.
    pub indexed_at: DateTime<Utc>,
    /// Metadata.
    pub metadata: serde_json::Value,
    /// Chunks with embeddings (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<ChunkExport>>,
}

/// Chunk export record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkExport {
    /// Chunk ID.
    pub id: String,
    /// Chunk text.
    pub text: String,
    /// Start offset in document.
    pub start_offset: usize,
    /// End offset in document.
    pub end_offset: usize,
    /// Embedding vector (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// Backup manager for creating and restoring backups.
pub struct BackupManager {
    /// Data directory for backups.
    backup_dir: PathBuf,
}

impl BackupManager {
    /// Create a new backup manager.
    pub fn new(backup_dir: impl AsRef<Path>) -> Result<Self> {
        let backup_dir = backup_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&backup_dir)?;
        Ok(Self { backup_dir })
    }

    /// Get the backup directory.
    pub fn backup_dir(&self) -> &Path {
        &self.backup_dir
    }

    /// List available backups.
    pub fn list_backups(&self) -> Result<Vec<BackupMetadata>> {
        let mut backups = Vec::new();

        for entry in std::fs::read_dir(&self.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&content) {
                        backups.push(metadata);
                    }
                }
            }
        }

        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(backups)
    }

    /// Create a backup of the index.
    ///
    /// This creates a portable backup containing all documents and their embeddings.
    pub async fn create_backup(
        &self,
        storage: &dyn StorageBackend,
        storage_backend: &str,
        embedding_dimension: usize,
        description: Option<String>,
    ) -> Result<BackupResult> {
        let start = std::time::Instant::now();

        // Get stats for metadata
        let stats = storage.stats().await?;

        // Create metadata
        let mut metadata = BackupMetadata::new(
            stats.document_count,
            stats.chunk_count,
            storage_backend,
            embedding_dimension,
        );
        if let Some(desc) = description {
            metadata = metadata.with_description(desc);
        }

        // Create backup file path
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_filename = format!("backup_{}.jsonl", timestamp);
        let backup_path = self.backup_dir.join(&backup_filename);
        let metadata_path = self.backup_dir.join(format!("backup_{}.json", timestamp));

        // Export all documents with embeddings
        let options = ExportOptions {
            format: ExportFormat::Jsonl,
            include_content: true,
            include_embeddings: true,
            source_id: None,
            compress: false,
        };

        let export_result = self
            .export_documents(storage, &backup_path, &options)
            .await?;

        // Update metadata with actual size
        metadata.size_bytes = export_result.size_bytes;

        // Write metadata file
        let metadata_file = File::create(&metadata_path)?;
        serde_json::to_writer_pretty(metadata_file, &metadata)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(BackupResult {
            metadata,
            path: backup_path,
            duration_ms,
        })
    }

    /// Restore from a backup.
    pub async fn restore_backup(
        &self,
        storage: &dyn StorageBackend,
        backup_path: impl AsRef<Path>,
    ) -> Result<RestoreResult> {
        let start = std::time::Instant::now();
        let backup_path = backup_path.as_ref();

        // Find and read metadata
        let metadata_path = backup_path.with_extension("json");
        let metadata: BackupMetadata = if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path)?;
            serde_json::from_str(&content)?
        } else {
            // Create minimal metadata
            BackupMetadata::new(0, 0, "unknown", 384)
        };

        // Import documents
        let import_result = self.import_documents(storage, backup_path).await?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(RestoreResult {
            metadata,
            documents_restored: import_result.0,
            chunks_restored: import_result.1,
            duration_ms,
        })
    }

    /// Export documents to a file.
    pub async fn export_documents(
        &self,
        storage: &dyn StorageBackend,
        output_path: impl AsRef<Path>,
        options: &ExportOptions,
    ) -> Result<ExportResult> {
        let start = std::time::Instant::now();
        let output_path = output_path.as_ref().to_path_buf();

        // Get all documents
        let all_docs = storage
            .get_all_documents(options.source_id.as_deref())
            .await?;

        let file = File::create(&output_path)?;
        let mut writer = BufWriter::new(file);
        let mut document_count = 0;

        match options.format {
            ExportFormat::Jsonl => {
                for doc in all_docs {
                    let export = self.document_to_export(storage, doc, options).await?;
                    serde_json::to_writer(&mut writer, &export)?;
                    writer.write_all(b"\n")?;
                    document_count += 1;
                }
            }
            ExportFormat::Json => {
                let mut exports = Vec::new();
                for doc in all_docs {
                    let export = self.document_to_export(storage, doc, options).await?;
                    exports.push(export);
                    document_count += 1;
                }
                serde_json::to_writer_pretty(&mut writer, &exports)?;
            }
        }

        writer.flush()?;
        drop(writer);

        let size_bytes = std::fs::metadata(&output_path)?.len();
        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ExportResult {
            path: output_path,
            document_count,
            size_bytes,
            duration_ms,
        })
    }

    /// Convert a document to export format.
    async fn document_to_export(
        &self,
        storage: &dyn StorageBackend,
        doc: IndexedDocument,
        options: &ExportOptions,
    ) -> Result<DocumentExport> {
        let chunks = if options.include_embeddings {
            let doc_chunks = storage.get_document_chunks(&doc.id).await?;
            Some(
                doc_chunks
                    .into_iter()
                    .map(|c| ChunkExport {
                        id: c.id,
                        text: c.text,
                        start_offset: c.start_offset,
                        end_offset: c.end_offset,
                        embedding: Some(c.vector),
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(DocumentExport {
            id: doc.id,
            source_id: doc.source_id,
            path: doc.path,
            mime_type: doc.mime_type,
            size: doc.size,
            content: if options.include_content {
                Some(doc.content)
            } else {
                None
            },
            modified_at: doc.modified_at,
            indexed_at: doc.indexed_at,
            metadata: doc.metadata,
            chunks,
        })
    }

    /// Import documents from a file.
    pub async fn import_documents(
        &self,
        storage: &dyn StorageBackend,
        input_path: impl AsRef<Path>,
    ) -> Result<(usize, usize)> {
        let input_path = input_path.as_ref();
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);

        let mut documents_imported = 0;
        let mut chunks_imported = 0;

        // Try to detect format
        let extension = input_path.extension().and_then(|e| e.to_str());
        let is_json_array = extension == Some("json");

        if is_json_array {
            // Read as JSON array
            let mut content = String::new();
            let mut file = File::open(input_path)?;
            file.read_to_string(&mut content)?;

            let exports: Vec<DocumentExport> = serde_json::from_str(&content)?;
            for export in exports {
                let (doc, chunks) = self.export_to_document(export)?;
                storage.store(doc, chunks.clone()).await?;
                documents_imported += 1;
                chunks_imported += chunks.len();
            }
        } else {
            // Read as JSONL
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                let export: DocumentExport = serde_json::from_str(&line)?;
                let (doc, chunks) = self.export_to_document(export)?;
                storage.store(doc, chunks.clone()).await?;
                documents_imported += 1;
                chunks_imported += chunks.len();
            }
        }

        Ok((documents_imported, chunks_imported))
    }

    /// Convert export format to document and chunks.
    fn export_to_document(
        &self,
        export: DocumentExport,
    ) -> Result<(IndexedDocument, Vec<VectorChunk>)> {
        let doc = IndexedDocument {
            id: export.id.clone(),
            source_id: export.source_id,
            path: export.path,
            mime_type: export.mime_type,
            size: export.size,
            content: export.content.unwrap_or_default(),
            modified_at: export.modified_at,
            indexed_at: export.indexed_at,
            metadata: export.metadata,
        };

        let chunks = export
            .chunks
            .unwrap_or_default()
            .into_iter()
            .map(|c| VectorChunk {
                id: c.id,
                document_id: export.id.clone(),
                text: c.text,
                vector: c.embedding.unwrap_or_default(),
                start_offset: c.start_offset,
                end_offset: c.end_offset,
            })
            .collect();

        Ok((doc, chunks))
    }

    /// Delete a backup.
    pub fn delete_backup(&self, backup_id: &str) -> Result<bool> {
        // Find backup files by ID
        for entry in std::fs::read_dir(&self.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&content) {
                        if metadata.backup_id == backup_id {
                            // Delete metadata file
                            std::fs::remove_file(&path)?;

                            // Delete data file
                            let data_path = path.with_extension("jsonl");
                            if data_path.exists() {
                                std::fs::remove_file(&data_path)?;
                            }

                            return Ok(true);
                        }
                    }
                }
            }
        }

        Ok(false)
    }
}

/// Import result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// Number of documents imported.
    pub documents_imported: usize,
    /// Number of chunks imported.
    pub chunks_imported: usize,
    /// Duration of the import operation.
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_backup_metadata() {
        let metadata =
            BackupMetadata::new(100, 500, "embedded", 384).with_description("Test backup");

        assert_eq!(metadata.document_count, 100);
        assert_eq!(metadata.chunk_count, 500);
        assert_eq!(metadata.description, Some("Test backup".to_string()));
    }

    #[test]
    fn test_export_options_default() {
        let options = ExportOptions::default();
        assert_eq!(options.format, ExportFormat::Jsonl);
        assert!(options.include_content);
        assert!(!options.include_embeddings);
    }

    #[test]
    fn test_backup_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path());
        assert!(manager.is_ok());
    }

    #[test]
    fn test_list_empty_backups() {
        let temp_dir = TempDir::new().unwrap();
        let manager = BackupManager::new(temp_dir.path()).unwrap();
        let backups = manager.list_backups().unwrap();
        assert!(backups.is_empty());
    }

    #[test]
    fn test_document_export_serialization() {
        let export = DocumentExport {
            id: "doc1".to_string(),
            source_id: "source1".to_string(),
            path: "/test/doc.txt".to_string(),
            mime_type: "text/plain".to_string(),
            size: 100,
            content: Some("Test content".to_string()),
            modified_at: Utc::now(),
            indexed_at: Utc::now(),
            metadata: serde_json::json!({}),
            chunks: None,
        };

        let json = serde_json::to_string(&export).unwrap();
        let parsed: DocumentExport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, export.id);
        assert_eq!(parsed.content, export.content);
    }
}
