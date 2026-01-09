//! Tests for the IndexCoordinator.

use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

use alloy::config::Config;
use alloy::coordinator::IndexCoordinator;
use alloy::search::HybridQuery;

/// Create a test configuration with embedded storage.
fn create_test_config(data_dir: &std::path::Path) -> Config {
    let mut config = Config::default();
    config.storage.data_dir = data_dir.to_string_lossy().to_string();
    config
}

/// Create test files in a directory.
fn create_test_files(dir: &TempDir) -> Vec<std::path::PathBuf> {
    let files = vec![
        (
            "doc1.txt",
            "This is a document about machine learning and neural networks.",
        ),
        (
            "doc2.txt",
            "Natural language processing is a fascinating field of AI.",
        ),
        (
            "doc3.md",
            "# Rust Programming\n\nRust is a systems programming language focused on safety.",
        ),
    ];

    let mut paths = Vec::new();
    for (name, content) in files {
        let path = dir.path().join(name);
        let mut f = File::create(&path).unwrap();
        writeln!(f, "{}", content).unwrap();
        paths.push(path);
    }
    paths
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_coordinator_initialization() {
    let data_dir = TempDir::new().unwrap();
    let config = create_test_config(data_dir.path());

    let coordinator = IndexCoordinator::new(config).await;
    assert!(
        coordinator.is_ok(),
        "Coordinator should initialize successfully"
    );
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_index_local_directory() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create test files
    create_test_files(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Index the directory
    let result = coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["*.txt".to_string(), "*.md".to_string()],
            vec![],
            false,
        )
        .await;

    assert!(result.is_ok(), "Indexing should succeed");
    let source = result.unwrap();
    assert_eq!(source.document_count, 3, "Should index all 3 documents");
    assert_eq!(source.source_type, "local");
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_search_indexed_documents() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create test files
    create_test_files(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Index the directory
    coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["*.txt".to_string(), "*.md".to_string()],
            vec![],
            false,
        )
        .await
        .unwrap();

    // Search for machine learning
    let query = HybridQuery::new("machine learning neural networks")
        .limit(10)
        .vector_weight(0.5);

    let results = coordinator.search(query).await;
    assert!(results.is_ok(), "Search should succeed");

    let response = results.unwrap();
    assert!(
        !response.results.is_empty(),
        "Should find at least one result"
    );

    // The first result should be doc1.txt which mentions machine learning
    let first_result = &response.results[0];
    assert!(
        first_result.text.contains("machine learning") || first_result.text.contains("neural"),
        "First result should be relevant to the query"
    );
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_list_sources() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    create_test_files(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Initially no sources
    let sources = coordinator.list_sources().await;
    assert!(sources.is_empty(), "Should have no sources initially");

    // Index a directory
    coordinator
        .index_local(content_dir.path().to_path_buf(), vec![], vec![], false)
        .await
        .unwrap();

    // Should have one source now
    let sources = coordinator.list_sources().await;
    assert_eq!(sources.len(), 1, "Should have one source");
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_get_document() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create a single file
    let file_path = content_dir.path().join("test.txt");
    let mut f = File::create(&file_path).unwrap();
    writeln!(f, "This is test content for document retrieval.").unwrap();

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Index
    coordinator
        .index_local(
            content_dir.path().to_path_buf(),
            vec!["*.txt".to_string()],
            vec![],
            false,
        )
        .await
        .unwrap();

    // Get stats to verify documents exist
    let stats = coordinator.stats().await.unwrap();
    assert!(
        stats.document_count > 0,
        "Should have at least one document"
    );
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_storage_stats() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    create_test_files(&content_dir);

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    // Check initial stats
    let initial_stats = coordinator.stats().await.unwrap();
    assert_eq!(
        initial_stats.document_count, 0,
        "Should start with 0 documents"
    );
    assert_eq!(initial_stats.chunk_count, 0, "Should start with 0 chunks");

    // Index files
    coordinator
        .index_local(content_dir.path().to_path_buf(), vec![], vec![], false)
        .await
        .unwrap();

    // Check updated stats
    let stats = coordinator.stats().await.unwrap();
    assert!(
        stats.document_count > 0,
        "Should have documents after indexing"
    );
    assert!(stats.chunk_count > 0, "Should have chunks after indexing");
}

#[tokio::test]
#[ignore = "requires embedding model download - run with cargo test -- --ignored"]
async fn test_hybrid_search_weights() {
    let content_dir = TempDir::new().unwrap();
    let data_dir = TempDir::new().unwrap();

    // Create files with specific keywords
    let file1 = content_dir.path().join("exact.txt");
    let mut f = File::create(&file1).unwrap();
    writeln!(f, "exact keyword match for testing").unwrap();

    let file2 = content_dir.path().join("semantic.txt");
    let mut f = File::create(&file2).unwrap();
    writeln!(f, "precise term correspondence for evaluation purposes").unwrap();

    let config = create_test_config(data_dir.path());
    let coordinator = IndexCoordinator::new(config).await.unwrap();

    coordinator
        .index_local(content_dir.path().to_path_buf(), vec![], vec![], false)
        .await
        .unwrap();

    // Test with high text weight (keyword focused)
    let keyword_query = HybridQuery::new("exact keyword match")
        .limit(5)
        .vector_weight(0.1); // Low vector weight = high text weight

    let keyword_results = coordinator.search(keyword_query).await.unwrap();
    assert!(!keyword_results.results.is_empty());

    // Test with high vector weight (semantic focused)
    let semantic_query = HybridQuery::new("precise correspondence")
        .limit(5)
        .vector_weight(0.9); // High vector weight

    let semantic_results = coordinator.search(semantic_query).await.unwrap();
    assert!(!semantic_results.results.is_empty());
}
