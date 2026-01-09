//! Document deduplication module.
//!
//! Provides multiple strategies for detecting duplicate content:
//! - Exact: SHA-256 content hash matching
//! - MinHash: Locality-sensitive hashing for near-duplicate detection
//! - Semantic: Embedding-based similarity threshold

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Result of a deduplication check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationResult {
    /// Whether this content is a duplicate of an existing document.
    pub is_duplicate: bool,
    /// ID of the original document if this is a duplicate.
    pub duplicate_of: Option<String>,
    /// Similarity score (1.0 for exact match, 0.0-1.0 for fuzzy).
    pub similarity: f32,
    /// Strategy that detected the duplicate.
    pub strategy: DeduplicationStrategy,
}

/// Deduplication strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeduplicationStrategy {
    /// Exact content hash matching.
    Exact,
    /// MinHash locality-sensitive hashing.
    MinHash,
    /// Embedding cosine similarity.
    Semantic,
}

/// Action to take when a duplicate is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeduplicationAction {
    /// Skip indexing duplicates.
    Skip,
    /// Index with reference to original.
    Flag,
    /// Update the existing document.
    Update,
}

/// Deduplication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeduplicationConfig {
    /// Enable deduplication.
    pub enabled: bool,
    /// Primary strategy to use.
    pub strategy: DeduplicationStrategy,
    /// Similarity threshold for fuzzy matching (0.0-1.0).
    pub threshold: f32,
    /// Action to take when duplicate is detected.
    pub action: DeduplicationAction,
    /// Number of hash functions for MinHash (higher = more accurate but slower).
    pub minhash_num_hashes: usize,
    /// Shingle size for MinHash (number of consecutive tokens).
    pub shingle_size: usize,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: DeduplicationStrategy::Exact,
            threshold: 0.85,
            action: DeduplicationAction::Skip,
            minhash_num_hashes: 128,
            shingle_size: 3,
        }
    }
}

/// Trait for deduplication implementations.
#[async_trait]
pub trait Deduplicator: Send + Sync {
    /// Check if content is a duplicate of an existing document.
    async fn check(&self, content: &str, doc_id: &str)
        -> crate::error::Result<DeduplicationResult>;

    /// Register content for future duplicate checking.
    async fn register(&self, content: &str, doc_id: &str) -> crate::error::Result<()>;

    /// Remove a document from the deduplication index.
    async fn remove(&self, doc_id: &str) -> crate::error::Result<()>;

    /// Clear all registered documents.
    async fn clear(&self) -> crate::error::Result<()>;
}

/// Exact hash-based deduplicator using SHA-256.
pub struct ExactDeduplicator {
    /// Map of content hash to document ID.
    hashes: RwLock<HashMap<String, String>>,
}

impl ExactDeduplicator {
    /// Create a new exact deduplicator.
    pub fn new() -> Self {
        Self {
            hashes: RwLock::new(HashMap::new()),
        }
    }

    /// Compute SHA-256 hash of content.
    fn compute_hash(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

impl Default for ExactDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Deduplicator for ExactDeduplicator {
    async fn check(
        &self,
        content: &str,
        _doc_id: &str,
    ) -> crate::error::Result<DeduplicationResult> {
        let hash = Self::compute_hash(content);
        let hashes = self.hashes.read().await;

        if let Some(original_id) = hashes.get(&hash) {
            Ok(DeduplicationResult {
                is_duplicate: true,
                duplicate_of: Some(original_id.clone()),
                similarity: 1.0,
                strategy: DeduplicationStrategy::Exact,
            })
        } else {
            Ok(DeduplicationResult {
                is_duplicate: false,
                duplicate_of: None,
                similarity: 0.0,
                strategy: DeduplicationStrategy::Exact,
            })
        }
    }

    async fn register(&self, content: &str, doc_id: &str) -> crate::error::Result<()> {
        let hash = Self::compute_hash(content);
        let mut hashes = self.hashes.write().await;
        hashes.insert(hash, doc_id.to_string());
        Ok(())
    }

    async fn remove(&self, doc_id: &str) -> crate::error::Result<()> {
        let mut hashes = self.hashes.write().await;
        hashes.retain(|_, v| v != doc_id);
        Ok(())
    }

    async fn clear(&self) -> crate::error::Result<()> {
        let mut hashes = self.hashes.write().await;
        hashes.clear();
        Ok(())
    }
}

/// MinHash signature for near-duplicate detection.
#[derive(Clone)]
struct MinHashSignature {
    /// The minimum hash values for each hash function.
    hashes: Vec<u64>,
}

impl MinHashSignature {
    /// Create a new MinHash signature from text content.
    fn new(content: &str, num_hashes: usize, shingle_size: usize) -> Self {
        let shingles = Self::get_shingles(content, shingle_size);
        let mut hashes = vec![u64::MAX; num_hashes];

        for shingle in shingles {
            for (i, hash) in hashes.iter_mut().enumerate() {
                let h = Self::hash_shingle(&shingle, i as u64);
                if h < *hash {
                    *hash = h;
                }
            }
        }

        Self { hashes }
    }

    /// Split text into shingles (n-grams of words).
    fn get_shingles(content: &str, size: usize) -> Vec<String> {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() < size {
            return vec![content.to_lowercase()];
        }

        words
            .windows(size)
            .map(|w| w.join(" ").to_lowercase())
            .collect()
    }

    /// Hash a shingle with a specific seed.
    fn hash_shingle(shingle: &str, seed: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        shingle.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute Jaccard similarity between two signatures.
    fn similarity(&self, other: &Self) -> f32 {
        if self.hashes.len() != other.hashes.len() {
            return 0.0;
        }

        let matching = self
            .hashes
            .iter()
            .zip(other.hashes.iter())
            .filter(|(a, b)| a == b)
            .count();

        matching as f32 / self.hashes.len() as f32
    }
}

/// MinHash-based deduplicator for near-duplicate detection.
pub struct MinHashDeduplicator {
    /// Configuration.
    config: DeduplicationConfig,
    /// Map of document ID to MinHash signature.
    signatures: RwLock<HashMap<String, MinHashSignature>>,
}

impl MinHashDeduplicator {
    /// Create a new MinHash deduplicator.
    pub fn new(config: DeduplicationConfig) -> Self {
        Self {
            config,
            signatures: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Deduplicator for MinHashDeduplicator {
    async fn check(
        &self,
        content: &str,
        _doc_id: &str,
    ) -> crate::error::Result<DeduplicationResult> {
        let new_sig = MinHashSignature::new(
            content,
            self.config.minhash_num_hashes,
            self.config.shingle_size,
        );
        let signatures = self.signatures.read().await;

        let mut best_match: Option<(String, f32)> = None;

        for (id, sig) in signatures.iter() {
            let similarity = new_sig.similarity(sig);
            if similarity >= self.config.threshold
                && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
            {
                best_match = Some((id.clone(), similarity));
            }
        }

        if let Some((original_id, similarity)) = best_match {
            Ok(DeduplicationResult {
                is_duplicate: true,
                duplicate_of: Some(original_id),
                similarity,
                strategy: DeduplicationStrategy::MinHash,
            })
        } else {
            Ok(DeduplicationResult {
                is_duplicate: false,
                duplicate_of: None,
                similarity: 0.0,
                strategy: DeduplicationStrategy::MinHash,
            })
        }
    }

    async fn register(&self, content: &str, doc_id: &str) -> crate::error::Result<()> {
        let sig = MinHashSignature::new(
            content,
            self.config.minhash_num_hashes,
            self.config.shingle_size,
        );
        let mut signatures = self.signatures.write().await;
        signatures.insert(doc_id.to_string(), sig);
        Ok(())
    }

    async fn remove(&self, doc_id: &str) -> crate::error::Result<()> {
        let mut signatures = self.signatures.write().await;
        signatures.remove(doc_id);
        Ok(())
    }

    async fn clear(&self) -> crate::error::Result<()> {
        let mut signatures = self.signatures.write().await;
        signatures.clear();
        Ok(())
    }
}

/// Semantic deduplicator using embedding similarity.
pub struct SemanticDeduplicator {
    /// Configuration.
    config: DeduplicationConfig,
    /// Embedding provider.
    embedder: Arc<dyn crate::embedding::EmbeddingProvider>,
    /// Map of document ID to embedding vector.
    embeddings: RwLock<HashMap<String, Vec<f32>>>,
}

impl SemanticDeduplicator {
    /// Create a new semantic deduplicator.
    pub fn new(
        config: DeduplicationConfig,
        embedder: Arc<dyn crate::embedding::EmbeddingProvider>,
    ) -> Self {
        Self {
            config,
            embedder,
            embeddings: RwLock::new(HashMap::new()),
        }
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

#[async_trait]
impl Deduplicator for SemanticDeduplicator {
    async fn check(
        &self,
        content: &str,
        _doc_id: &str,
    ) -> crate::error::Result<DeduplicationResult> {
        // Generate embedding for the new content
        let new_embedding = self.embedder.embed(&[content.to_string()]).await?;
        if new_embedding.is_empty() {
            return Ok(DeduplicationResult {
                is_duplicate: false,
                duplicate_of: None,
                similarity: 0.0,
                strategy: DeduplicationStrategy::Semantic,
            });
        }
        let new_vec = &new_embedding[0];

        let embeddings = self.embeddings.read().await;

        let mut best_match: Option<(String, f32)> = None;

        for (id, vec) in embeddings.iter() {
            let similarity = Self::cosine_similarity(new_vec, vec);
            if similarity >= self.config.threshold
                && (best_match.is_none() || similarity > best_match.as_ref().unwrap().1)
            {
                best_match = Some((id.clone(), similarity));
            }
        }

        if let Some((original_id, similarity)) = best_match {
            Ok(DeduplicationResult {
                is_duplicate: true,
                duplicate_of: Some(original_id),
                similarity,
                strategy: DeduplicationStrategy::Semantic,
            })
        } else {
            Ok(DeduplicationResult {
                is_duplicate: false,
                duplicate_of: None,
                similarity: 0.0,
                strategy: DeduplicationStrategy::Semantic,
            })
        }
    }

    async fn register(&self, content: &str, doc_id: &str) -> crate::error::Result<()> {
        let embedding = self.embedder.embed(&[content.to_string()]).await?;
        if !embedding.is_empty() {
            let mut embeddings = self.embeddings.write().await;
            embeddings.insert(doc_id.to_string(), embedding[0].clone());
        }
        Ok(())
    }

    async fn remove(&self, doc_id: &str) -> crate::error::Result<()> {
        let mut embeddings = self.embeddings.write().await;
        embeddings.remove(doc_id);
        Ok(())
    }

    async fn clear(&self) -> crate::error::Result<()> {
        let mut embeddings = self.embeddings.write().await;
        embeddings.clear();
        Ok(())
    }
}

/// Composite deduplicator that combines multiple strategies.
pub struct CompositeDeduplicator {
    /// Deduplicators to use in order (first match wins).
    deduplicators: Vec<Box<dyn Deduplicator>>,
}

impl CompositeDeduplicator {
    /// Create a new composite deduplicator.
    pub fn new(deduplicators: Vec<Box<dyn Deduplicator>>) -> Self {
        Self { deduplicators }
    }

    /// Create from configuration and optional embedding provider.
    pub fn from_config(
        config: &DeduplicationConfig,
        embedder: Option<Arc<dyn crate::embedding::EmbeddingProvider>>,
    ) -> Self {
        let mut deduplicators: Vec<Box<dyn Deduplicator>> = Vec::new();

        match config.strategy {
            DeduplicationStrategy::Exact => {
                deduplicators.push(Box::new(ExactDeduplicator::new()));
            }
            DeduplicationStrategy::MinHash => {
                deduplicators.push(Box::new(MinHashDeduplicator::new(config.clone())));
            }
            DeduplicationStrategy::Semantic => {
                if let Some(emb) = embedder {
                    deduplicators.push(Box::new(SemanticDeduplicator::new(config.clone(), emb)));
                } else {
                    // Fall back to MinHash if no embedder available
                    deduplicators.push(Box::new(MinHashDeduplicator::new(config.clone())));
                }
            }
        }

        Self { deduplicators }
    }
}

#[async_trait]
impl Deduplicator for CompositeDeduplicator {
    async fn check(
        &self,
        content: &str,
        doc_id: &str,
    ) -> crate::error::Result<DeduplicationResult> {
        for dedup in &self.deduplicators {
            let result = dedup.check(content, doc_id).await?;
            if result.is_duplicate {
                return Ok(result);
            }
        }

        Ok(DeduplicationResult {
            is_duplicate: false,
            duplicate_of: None,
            similarity: 0.0,
            strategy: DeduplicationStrategy::Exact,
        })
    }

    async fn register(&self, content: &str, doc_id: &str) -> crate::error::Result<()> {
        for dedup in &self.deduplicators {
            dedup.register(content, doc_id).await?;
        }
        Ok(())
    }

    async fn remove(&self, doc_id: &str) -> crate::error::Result<()> {
        for dedup in &self.deduplicators {
            dedup.remove(doc_id).await?;
        }
        Ok(())
    }

    async fn clear(&self) -> crate::error::Result<()> {
        for dedup in &self.deduplicators {
            dedup.clear().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_deduplicator() {
        let dedup = ExactDeduplicator::new();
        let content = "This is a test document.";

        // Register document
        dedup.register(content, "doc1").await.unwrap();

        // Check same content - should be duplicate
        let result = dedup.check(content, "doc2").await.unwrap();
        assert!(result.is_duplicate);
        assert_eq!(result.duplicate_of, Some("doc1".to_string()));
        assert_eq!(result.similarity, 1.0);

        // Check different content - should not be duplicate
        let result = dedup.check("Different content", "doc3").await.unwrap();
        assert!(!result.is_duplicate);
    }

    #[tokio::test]
    async fn test_minhash_deduplicator() {
        let config = DeduplicationConfig {
            enabled: true,
            strategy: DeduplicationStrategy::MinHash,
            threshold: 0.5, // Lower threshold for testing
            minhash_num_hashes: 64,
            shingle_size: 2,
            ..Default::default()
        };
        let dedup = MinHashDeduplicator::new(config);

        let content1 = "The quick brown fox jumps over the lazy dog near the river bank";
        let content2 = "The quick brown fox jumps over the lazy cat near the river bank"; // Similar

        // Register first document
        dedup.register(content1, "doc1").await.unwrap();

        // Check similar content - should be duplicate
        let result = dedup.check(content2, "doc2").await.unwrap();
        assert!(result.is_duplicate);
        assert!(result.similarity > 0.5);

        // Check very different content
        let result = dedup
            .check("Completely unrelated text about space exploration", "doc3")
            .await
            .unwrap();
        assert!(!result.is_duplicate);
    }

    #[tokio::test]
    async fn test_minhash_signature_similarity() {
        let sig1 = MinHashSignature::new("The quick brown fox jumps over the lazy dog", 128, 3);
        let sig2 = MinHashSignature::new("The quick brown fox jumps over the lazy cat", 128, 3);
        let sig3 =
            MinHashSignature::new("A completely different sentence about programming", 128, 3);

        // Similar sentences should have high similarity
        let sim_12 = sig1.similarity(&sig2);
        assert!(
            sim_12 > 0.5,
            "Similar sentences should have similarity > 0.5, got {}",
            sim_12
        );

        // Different sentences should have low similarity
        let sim_13 = sig1.similarity(&sig3);
        assert!(
            sim_13 < 0.3,
            "Different sentences should have similarity < 0.3, got {}",
            sim_13
        );
    }

    #[tokio::test]
    async fn test_exact_deduplicator_remove() {
        let dedup = ExactDeduplicator::new();
        let content = "Test content for removal";

        // Register and verify
        dedup.register(content, "doc1").await.unwrap();
        let result = dedup.check(content, "doc2").await.unwrap();
        assert!(result.is_duplicate);

        // Remove and verify
        dedup.remove("doc1").await.unwrap();
        let result = dedup.check(content, "doc2").await.unwrap();
        assert!(!result.is_duplicate);
    }

    #[tokio::test]
    async fn test_deduplicator_clear() {
        let dedup = ExactDeduplicator::new();

        // Register multiple documents
        dedup.register("Content 1", "doc1").await.unwrap();
        dedup.register("Content 2", "doc2").await.unwrap();

        // Clear and verify
        dedup.clear().await.unwrap();
        let result = dedup.check("Content 1", "doc3").await.unwrap();
        assert!(!result.is_duplicate);
    }
}
