//! Tantivy full-text search backend.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{document::Value, Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::error::{Result, StorageError};
use crate::storage::IndexedDocument;

/// Result from a full-text search.
#[derive(Debug, Clone)]
pub struct FullTextResult {
    /// Document ID
    pub document_id: String,
    /// BM25 score
    pub score: f32,
    /// Matched text snippet
    pub snippet: String,
}

/// Tantivy full-text search backend.
pub struct TantivyBackend {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    doc_id_field: Field,
    source_id_field: Field,
    path_field: Field,
    content_field: Field,
    mime_type_field: Field,
}

impl TantivyBackend {
    /// Create a new Tantivy backend.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let index_path = data_dir.join("tantivy");
        std::fs::create_dir_all(&index_path).map_err(StorageError::Io)?;

        // Build schema
        let mut schema_builder = Schema::builder();
        let doc_id_field = schema_builder.add_text_field("doc_id", STRING | STORED);
        let source_id_field = schema_builder.add_text_field("source_id", STRING | STORED);
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let mime_type_field = schema_builder.add_text_field("mime_type", STRING | STORED);
        let schema = schema_builder.build();

        // Open or create index
        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(&index_path).map_err(|e| StorageError::Index(e.to_string()))?
        } else {
            Index::create_in_dir(&index_path, schema.clone())
                .map_err(|e| StorageError::Index(e.to_string()))?
        };

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: tantivy::TantivyError| StorageError::Index(e.to_string()))?;

        // Create writer with 50MB buffer
        let writer = index
            .writer(50_000_000)
            .map_err(|e| StorageError::Index(e.to_string()))?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            schema,
            doc_id_field,
            source_id_field,
            path_field,
            content_field,
            mime_type_field,
        })
    }

    /// Index a document.
    pub fn index_document(&self, doc: &IndexedDocument) -> Result<()> {
        let writer = self.writer.write();

        // Delete existing document with same ID
        let term = tantivy::Term::from_field_text(self.doc_id_field, &doc.id);
        writer.delete_term(term);

        // Add new document
        let tantivy_doc = doc!(
            self.doc_id_field => doc.id.clone(),
            self.source_id_field => doc.source_id.clone(),
            self.path_field => doc.path.clone(),
            self.content_field => doc.content.clone(),
            self.mime_type_field => doc.mime_type.clone(),
        );

        writer
            .add_document(tantivy_doc)
            .map_err(|e| StorageError::Index(e.to_string()))?;

        Ok(())
    }

    /// Commit pending changes.
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write();
        writer
            .commit()
            .map_err(|e| StorageError::Index(e.to_string()))?;

        // Reload the reader to see the new changes
        self.reader
            .reload()
            .map_err(|e| StorageError::Index(e.to_string()))?;

        Ok(())
    }

    /// Search for documents.
    pub fn search(
        &self,
        query_text: &str,
        source_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<FullTextResult>> {
        let searcher = self.reader.searcher();

        // Build query
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        // Escape special characters in query
        let escaped_query = Self::escape_query(query_text);

        let query = query_parser
            .parse_query(&escaped_query)
            .map_err(|e| StorageError::Query(e.to_string()))?;

        // Execute search
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit * 2)) // Get extra to filter
            .map_err(|e| StorageError::Query(e.to_string()))?;

        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| StorageError::Query(e.to_string()))?;

            // Get document ID
            let doc_id = doc
                .get_first(self.doc_id_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            // Filter by source_id if specified
            if let Some(sid) = source_id {
                let doc_source_id = doc
                    .get_first(self.source_id_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if doc_source_id != sid {
                    continue;
                }
            }

            // Get content for snippet
            let content = doc
                .get_first(self.content_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            // Create a simple snippet (first 200 chars or content around match)
            let snippet = Self::create_snippet(content, query_text, 200);

            results.push(FullTextResult {
                document_id: doc_id,
                score,
                snippet,
            });

            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    /// Remove a document by ID.
    pub fn remove(&self, doc_id: &str) -> Result<()> {
        let writer = self.writer.write();
        let term = tantivy::Term::from_field_text(self.doc_id_field, doc_id);
        writer.delete_term(term);
        Ok(())
    }

    /// Get the number of documents.
    pub fn document_count(&self) -> Result<usize> {
        let searcher = self.reader.searcher();
        Ok(searcher.num_docs() as usize)
    }

    /// Escape special query characters.
    fn escape_query(query: &str) -> String {
        // Escape Tantivy special characters
        let special_chars = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/'];
        let mut escaped = String::with_capacity(query.len() * 2);

        for c in query.chars() {
            if special_chars.contains(&c) {
                escaped.push('\\');
            }
            escaped.push(c);
        }

        escaped
    }

    /// Create a snippet from content around query terms.
    fn create_snippet(content: &str, query: &str, max_len: usize) -> String {
        let content_lower = content.to_lowercase();
        let query_lower = query.to_lowercase();

        // Find first occurrence of any query word
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let mut best_pos = None;

        for word in &query_words {
            if let Some(pos) = content_lower.find(word) {
                if best_pos.is_none() || pos < best_pos.unwrap() {
                    best_pos = Some(pos);
                }
            }
        }

        match best_pos {
            Some(pos) => {
                // Start a bit before the match
                let start = pos.saturating_sub(50);
                let end = (start + max_len).min(content.len());

                // Find word boundaries
                let actual_start = content[..start].rfind(' ').map(|p| p + 1).unwrap_or(start);
                let actual_end = content[end..].find(' ').map(|p| end + p).unwrap_or(end);

                let mut snippet = String::new();
                if actual_start > 0 {
                    snippet.push_str("...");
                }
                snippet.push_str(content[actual_start..actual_end].trim());
                if actual_end < content.len() {
                    snippet.push_str("...");
                }
                snippet
            }
            None => {
                // No match found, return beginning of content
                if content.len() <= max_len {
                    content.to_string()
                } else {
                    let end = content[..max_len].rfind(' ').unwrap_or(max_len);
                    format!("{}...", &content[..end])
                }
            }
        }
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

    #[test]
    fn test_tantivy_backend_create() {
        let temp_dir = TempDir::new().unwrap();
        let backend = TantivyBackend::new(temp_dir.path());
        assert!(backend.is_ok());
    }

    #[test]
    fn test_tantivy_index_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let backend = TantivyBackend::new(temp_dir.path()).unwrap();

        // Index a document
        let doc = create_test_doc("doc1", "The quick brown fox jumps over the lazy dog");
        backend.index_document(&doc).unwrap();
        backend.commit().unwrap();

        // Search for it
        let results = backend.search("quick fox", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc1");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_tantivy_remove_document() {
        let temp_dir = TempDir::new().unwrap();
        let backend = TantivyBackend::new(temp_dir.path()).unwrap();

        // Index a document
        let doc = create_test_doc("doc1", "test content for removal");
        backend.index_document(&doc).unwrap();
        backend.commit().unwrap();

        // Verify it exists
        let results = backend.search("test content", None, 10).unwrap();
        assert_eq!(results.len(), 1);

        // Remove it
        backend.remove("doc1").unwrap();
        backend.commit().unwrap();

        // Verify it's gone
        let results = backend.search("test content", None, 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_tantivy_source_filter() {
        let temp_dir = TempDir::new().unwrap();
        let backend = TantivyBackend::new(temp_dir.path()).unwrap();

        // Index documents with different sources
        let mut doc1 = create_test_doc("doc1", "shared content");
        doc1.source_id = "source-a".to_string();

        let mut doc2 = create_test_doc("doc2", "shared content");
        doc2.source_id = "source-b".to_string();

        backend.index_document(&doc1).unwrap();
        backend.index_document(&doc2).unwrap();
        backend.commit().unwrap();

        // Search without filter
        let results = backend.search("shared content", None, 10).unwrap();
        assert_eq!(results.len(), 2);

        // Search with source filter
        let results = backend.search("shared content", Some("source-a"), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "doc1");
    }

    #[test]
    fn test_snippet_creation() {
        let content = "This is a long document with many words. The quick brown fox jumps over the lazy dog. More text follows here.";
        let snippet = TantivyBackend::create_snippet(content, "quick fox", 100);
        assert!(snippet.contains("quick"));
        assert!(snippet.contains("fox"));
    }
}
