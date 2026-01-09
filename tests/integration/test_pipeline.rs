//! End-to-end pipeline tests.

use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

use alloy::config::Config;
use alloy::coordinator::IndexCoordinator;
use alloy::search::HybridQuery;

/// Helper to create a test configuration.
fn create_test_config(data_dir: &std::path::Path) -> Config {
    let mut config = Config::default();
    config.storage.data_dir = data_dir.to_string_lossy().to_string();
    config
}

/// Create a diverse set of test documents.
fn create_diverse_documents(dir: &TempDir) {
    let documents = vec![
        (
            "readme.md",
            r#"
# Project README

This is a sample project demonstrating various capabilities.

## Features

- Fast indexing
- Hybrid search
- Multiple file formats

## Installation

Run `cargo install` to install.
"#,
        ),
        (
            "notes.txt",
            r#"
Meeting notes from 2024-01-15:

Discussed the new search feature implementation.
Key points:
- Need to support both vector and keyword search
- Performance should be under 100ms for most queries
- Support filtering by file type and date

Action items:
1. Design the hybrid search architecture
2. Implement embedding generation
3. Add full-text indexing
"#,
        ),
        (
            "code.json",
            r#"
{
  "name": "alloy",
  "version": "0.1.0",
  "description": "Hybrid document indexing MCP server",
  "dependencies": {
    "tantivy": "0.22",
    "lancedb": "0.19",
    "rmcp": "0.12"
  }
}
"#,
        ),
        (
            "config.yaml",
            r#"
server:
  host: localhost
  port: 8080

database:
  type: embedded
  path: ~/.local/share/alloy

search:
  vector_weight: 0.5
  default_limit: 10
"#,
        ),
    ];

    for (name, content) in documents {
        let path = dir.path().join(name);
        let mut f = File::create(&path).unwrap();
        write!(f, "{}", content).unwrap();
    }
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_full_pipeline_index_and_search() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    create_diverse_documents(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Step 1: Index all documents
    let source = coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["*".to_string()],
            vec![],
            false,
        )
        .await
        .unwrap();

    assert!(
        source.document_count >= 4,
        "Should index at least 4 documents"
    );

    // Step 2: Search for various queries
    let queries = vec![
        ("hybrid search", "Should find documents mentioning search"),
        ("installation", "Should find the README"),
        ("meeting notes", "Should find the notes file"),
        ("dependencies", "Should find the JSON file"),
    ];

    for (query_text, description) in queries {
        let query = HybridQuery::new(query_text).limit(5);
        let results = coordinator.search(query).await.unwrap();
        assert!(
            !results.results.is_empty(),
            "{}: query '{}' should return results",
            description,
            query_text
        );
    }

    // Step 3: Verify stats
    let stats = coordinator.stats().await.unwrap();
    assert!(stats.document_count > 0);
    assert!(stats.chunk_count > 0);
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_pipeline_with_filtering() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    create_diverse_documents(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Index documents
    let source = coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["*.md".to_string()], // Only markdown files
            vec![],
            false,
        )
        .await
        .unwrap();

    assert_eq!(
        source.document_count, 1,
        "Should only index the markdown file"
    );

    // Search should find content from the markdown file
    let query = HybridQuery::new("project features installation").limit(10);
    let results = coordinator.search(query).await.unwrap();

    assert!(!results.results.is_empty(), "Should find the README");
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_pipeline_reindexing() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create initial file
    let file1 = content_dir.path().join("doc.txt");
    let mut f = File::create(&file1).unwrap();
    writeln!(f, "Initial content about machine learning.").unwrap();

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Initial indexing
    let source1 = coordinator
        .index_local(content_dir.path().to_path_buf(), vec![], vec![], false)
        .await
        .unwrap();

    assert_eq!(source1.document_count, 1);

    // Add another file
    let file2 = content_dir.path().join("doc2.txt");
    let mut f = File::create(&file2).unwrap();
    writeln!(f, "New content about deep learning.").unwrap();

    // Re-index (with different source ID since we're creating a new source)
    let source2 = coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["doc2.txt".to_string()],
            vec![],
            false,
        )
        .await
        .unwrap();

    assert_eq!(source2.document_count, 1);

    // Verify both sources exist
    let sources = coordinator.list_sources().await;
    assert_eq!(sources.len(), 2, "Should have two sources now");
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_pipeline_large_documents() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create a larger document that will be chunked
    let file = content_dir.path().join("large.txt");
    let mut f = File::create(&file).unwrap();

    // Write multiple paragraphs
    for i in 0..20 {
        writeln!(
            f,
            "Paragraph {} about artificial intelligence and machine learning.",
            i
        )
        .unwrap();
        writeln!(
            f,
            "This discusses neural networks, deep learning, and natural language processing."
        )
        .unwrap();
        writeln!(
            f,
            "The field continues to advance with new architectures and training techniques."
        )
        .unwrap();
        writeln!(f).unwrap();
    }

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    let source = coordinator
        .index_local(content_dir.path().to_path_buf(), vec![], vec![], false)
        .await
        .unwrap();

    assert_eq!(source.document_count, 1);

    // Stats should show multiple chunks for the large document
    let stats = coordinator.stats().await.unwrap();
    assert!(
        stats.chunk_count > 1,
        "Large document should be split into multiple chunks"
    );

    // Search should work across chunks
    let query = HybridQuery::new("neural networks deep learning").limit(10);
    let results = coordinator.search(query).await.unwrap();
    assert!(
        !results.results.is_empty(),
        "Should find content in large document"
    );
}
