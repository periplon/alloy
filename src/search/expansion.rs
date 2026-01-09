//! Query expansion module for improving search recall.
//!
//! This module provides query expansion capabilities to find more relevant
//! documents by augmenting the original query with related terms.
//!
//! # Expansion Methods
//!
//! - **Embedding-based**: Find semantically similar terms using embeddings
//! - **Synonym-based**: Use string similarity and word stemming
//! - **Pseudo-relevance feedback**: Expand using terms from top results

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::QueryExpansionConfig;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;

/// Trait for query expansion implementations.
#[async_trait]
pub trait QueryExpander: Send + Sync {
    /// Expand a query with additional terms.
    async fn expand(&self, query: &str, config: &QueryExpansionConfig) -> Result<ExpandedQuery>;

    /// Get the name of this expander.
    fn name(&self) -> &str;
}

/// An expanded query with original and expansion terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedQuery {
    /// Original query text.
    pub original: String,
    /// Expanded query text (original + expansions).
    pub expanded: String,
    /// Individual expansion terms.
    pub expansion_terms: Vec<ExpansionTerm>,
    /// Whether expansion was applied.
    pub was_expanded: bool,
}

impl ExpandedQuery {
    /// Create a non-expanded query (passthrough).
    pub fn unchanged(query: &str) -> Self {
        Self {
            original: query.to_string(),
            expanded: query.to_string(),
            expansion_terms: vec![],
            was_expanded: false,
        }
    }

    /// Create an expanded query.
    pub fn with_expansions(query: &str, terms: Vec<ExpansionTerm>) -> Self {
        let expanded = if terms.is_empty() {
            query.to_string()
        } else {
            let expansion_text: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
            format!("{} {}", query, expansion_text.join(" "))
        };

        Self {
            original: query.to_string(),
            expanded,
            expansion_terms: terms.clone(),
            was_expanded: !terms.is_empty(),
        }
    }
}

/// A single expansion term with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionTerm {
    /// The expansion term.
    pub term: String,
    /// Similarity/confidence score (0.0 to 1.0).
    pub score: f32,
    /// Source of the expansion.
    pub source: ExpansionSource,
}

/// Source of an expansion term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpansionSource {
    /// From embedding similarity.
    Embedding,
    /// From synonym/string similarity.
    Synonym,
    /// From pseudo-relevance feedback.
    PseudoRelevance,
}

/// Embedding-based query expander.
///
/// Finds semantically similar terms by comparing query embeddings
/// with a vocabulary of known terms.
pub struct EmbeddingExpander {
    embedder: Arc<dyn EmbeddingProvider>,
    vocabulary: Vec<VocabEntry>,
}

struct VocabEntry {
    term: String,
    embedding: Vec<f32>,
}

impl EmbeddingExpander {
    /// Create a new embedding expander.
    pub fn new(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            embedder,
            vocabulary: Vec::new(),
        }
    }

    /// Add terms to the vocabulary for expansion.
    pub async fn add_vocabulary(&mut self, terms: &[String]) -> Result<()> {
        if terms.is_empty() {
            return Ok(());
        }

        let embeddings = self.embedder.embed(terms).await?;

        for (term, embedding) in terms.iter().zip(embeddings.into_iter()) {
            self.vocabulary.push(VocabEntry {
                term: term.clone(),
                embedding,
            });
        }

        Ok(())
    }

    /// Build vocabulary from document chunks.
    pub fn build_vocabulary_from_text(&mut self, texts: &[String]) {
        // Extract unique terms from texts
        let mut terms: HashSet<String> = HashSet::new();

        for text in texts {
            for word in text.split_whitespace() {
                let cleaned = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();

                if cleaned.len() >= 3 {
                    terms.insert(cleaned);
                }
            }
        }

        // Store terms (embeddings will be computed lazily)
        for term in terms {
            self.vocabulary.push(VocabEntry {
                term,
                embedding: vec![],
            });
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl QueryExpander for EmbeddingExpander {
    async fn expand(&self, query: &str, config: &QueryExpansionConfig) -> Result<ExpandedQuery> {
        if self.vocabulary.is_empty() {
            return Ok(ExpandedQuery::unchanged(query));
        }

        // Get query embedding
        let query_embeddings = self.embedder.embed(&[query.to_string()]).await?;
        let query_embedding = query_embeddings.into_iter().next().unwrap_or_default();

        if query_embedding.is_empty() {
            return Ok(ExpandedQuery::unchanged(query));
        }

        // Find similar terms from vocabulary
        let query_terms: HashSet<&str> = query.split_whitespace().collect();
        let mut candidates: Vec<ExpansionTerm> = Vec::new();

        for entry in &self.vocabulary {
            // Skip terms already in query
            if query_terms.contains(entry.term.as_str()) {
                continue;
            }

            // Skip if no embedding
            if entry.embedding.is_empty() {
                continue;
            }

            let similarity = Self::cosine_similarity(&query_embedding, &entry.embedding);

            if similarity >= config.similarity_threshold {
                candidates.push(ExpansionTerm {
                    term: entry.term.clone(),
                    score: similarity,
                    source: ExpansionSource::Embedding,
                });
            }
        }

        // Sort by similarity and take top expansions
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let expansions: Vec<ExpansionTerm> =
            candidates.into_iter().take(config.max_expansions).collect();

        Ok(ExpandedQuery::with_expansions(query, expansions))
    }

    fn name(&self) -> &str {
        "embedding"
    }
}

/// Synonym-based query expander using string similarity.
///
/// Uses edit distance and common affixes to find related terms.
pub struct SynonymExpander {
    synonyms: HashMap<String, Vec<String>>,
}

impl SynonymExpander {
    /// Create a new synonym expander with default synonyms.
    pub fn new() -> Self {
        Self {
            synonyms: Self::default_synonyms(),
        }
    }

    /// Create with custom synonyms.
    pub fn with_synonyms(synonyms: HashMap<String, Vec<String>>) -> Self {
        Self { synonyms }
    }

    /// Add synonyms for a term.
    pub fn add_synonyms(&mut self, term: &str, synonyms: Vec<String>) {
        self.synonyms.insert(term.to_lowercase(), synonyms);
    }

    /// Default technical synonyms.
    fn default_synonyms() -> HashMap<String, Vec<String>> {
        let mut map = HashMap::new();

        // Common technical synonyms
        map.insert(
            "error".into(),
            vec!["exception".into(), "failure".into(), "fault".into()],
        );
        map.insert(
            "create".into(),
            vec!["make".into(), "new".into(), "add".into(), "generate".into()],
        );
        map.insert(
            "delete".into(),
            vec!["remove".into(), "drop".into(), "erase".into()],
        );
        map.insert(
            "update".into(),
            vec!["modify".into(), "change".into(), "edit".into()],
        );
        map.insert(
            "find".into(),
            vec![
                "search".into(),
                "lookup".into(),
                "locate".into(),
                "query".into(),
            ],
        );
        map.insert(
            "config".into(),
            vec!["configuration".into(), "settings".into(), "options".into()],
        );
        map.insert(
            "auth".into(),
            vec![
                "authentication".into(),
                "authorization".into(),
                "login".into(),
            ],
        );
        map.insert(
            "api".into(),
            vec!["endpoint".into(), "service".into(), "interface".into()],
        );
        map.insert("db".into(), vec!["database".into(), "storage".into()]);
        map.insert(
            "async".into(),
            vec!["asynchronous".into(), "concurrent".into()],
        );
        map.insert("sync".into(), vec!["synchronous".into(), "blocking".into()]);

        map
    }

    /// Find similar terms using Levenshtein distance.
    fn find_similar(&self, term: &str, max_distance: usize) -> Vec<String> {
        let term_lower = term.to_lowercase();
        let mut similar = Vec::new();

        for key in self.synonyms.keys() {
            let distance = strsim::levenshtein(&term_lower, key);
            if distance <= max_distance && distance > 0 {
                similar.push(key.clone());
            }
        }

        similar
    }
}

impl Default for SynonymExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueryExpander for SynonymExpander {
    async fn expand(&self, query: &str, config: &QueryExpansionConfig) -> Result<ExpandedQuery> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let mut expansions: Vec<ExpansionTerm> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for term in &query_terms {
            let term_lower = term.to_lowercase();
            seen.insert(term_lower.clone());

            // Direct synonym lookup
            if let Some(syns) = self.synonyms.get(&term_lower) {
                for syn in syns {
                    if !seen.contains(syn) && expansions.len() < config.max_expansions {
                        seen.insert(syn.clone());
                        expansions.push(ExpansionTerm {
                            term: syn.clone(),
                            score: 0.9,
                            source: ExpansionSource::Synonym,
                        });
                    }
                }
            }

            // Find similar terms via edit distance
            let similar = self.find_similar(term, 2);
            for sim_term in similar {
                if let Some(syns) = self.synonyms.get(&sim_term) {
                    for syn in syns {
                        if !seen.contains(syn) && expansions.len() < config.max_expansions {
                            seen.insert(syn.clone());
                            expansions.push(ExpansionTerm {
                                term: syn.clone(),
                                score: 0.7,
                                source: ExpansionSource::Synonym,
                            });
                        }
                    }
                }
            }
        }

        Ok(ExpandedQuery::with_expansions(query, expansions))
    }

    fn name(&self) -> &str {
        "synonym"
    }
}

/// Hybrid expander combining embedding and synonym approaches.
pub struct HybridExpander {
    embedding_expander: EmbeddingExpander,
    synonym_expander: SynonymExpander,
}

impl HybridExpander {
    /// Create a new hybrid expander.
    pub fn new(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            embedding_expander: EmbeddingExpander::new(embedder),
            synonym_expander: SynonymExpander::new(),
        }
    }
}

#[async_trait]
impl QueryExpander for HybridExpander {
    async fn expand(&self, query: &str, config: &QueryExpansionConfig) -> Result<ExpandedQuery> {
        // Get expansions from both sources
        let synonym_result = self.synonym_expander.expand(query, config).await?;
        let embedding_result = self.embedding_expander.expand(query, config).await?;

        // Combine expansions, preferring embedding-based ones
        let mut all_terms: Vec<ExpansionTerm> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // Add synonym expansions first (they're more reliable)
        for term in synonym_result.expansion_terms {
            if !seen.contains(&term.term) {
                seen.insert(term.term.clone());
                all_terms.push(term);
            }
        }

        // Add embedding expansions
        for term in embedding_result.expansion_terms {
            if !seen.contains(&term.term) && all_terms.len() < config.max_expansions {
                seen.insert(term.term.clone());
                all_terms.push(term);
            }
        }

        // Sort by score
        all_terms.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_terms.truncate(config.max_expansions);

        Ok(ExpandedQuery::with_expansions(query, all_terms))
    }

    fn name(&self) -> &str {
        "hybrid"
    }
}

/// Pseudo-relevance feedback expander.
///
/// Expands queries using terms from top search results.
pub struct PseudoRelevanceFeedbackExpander {
    top_k: usize,
}

impl PseudoRelevanceFeedbackExpander {
    /// Create a new PRF expander.
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Extract expansion terms from result texts.
    pub fn extract_terms(&self, texts: &[String], query: &str) -> Vec<ExpansionTerm> {
        let query_terms: HashSet<String> =
            query.split_whitespace().map(|s| s.to_lowercase()).collect();

        // Count term frequencies across results
        let mut term_freq: HashMap<String, usize> = HashMap::new();

        for text in texts.iter().take(self.top_k) {
            for word in text.split_whitespace() {
                let cleaned = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();

                if cleaned.len() >= 3 && !query_terms.contains(&cleaned) {
                    *term_freq.entry(cleaned).or_insert(0) += 1;
                }
            }
        }

        // Convert to expansion terms, scored by frequency
        let max_freq = term_freq.values().copied().max().unwrap_or(1) as f32;
        let mut terms: Vec<ExpansionTerm> = term_freq
            .into_iter()
            .filter(|(_, freq)| *freq > 1) // Require appearing in multiple docs
            .map(|(term, freq)| ExpansionTerm {
                term,
                score: freq as f32 / max_freq,
                source: ExpansionSource::PseudoRelevance,
            })
            .collect();

        terms.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        terms
    }
}

/// Factory for creating query expanders based on configuration.
pub struct QueryExpanderFactory;

impl QueryExpanderFactory {
    /// Create a query expander based on configuration.
    pub fn create(
        config: &QueryExpansionConfig,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Box<dyn QueryExpander> {
        match config.method {
            crate::config::QueryExpansionMethod::Embedding => {
                Box::new(EmbeddingExpander::new(embedder))
            }
            crate::config::QueryExpansionMethod::Synonym => Box::new(SynonymExpander::new()),
            crate::config::QueryExpansionMethod::Hybrid => Box::new(HybridExpander::new(embedder)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expanded_query_unchanged() {
        let query = ExpandedQuery::unchanged("test query");
        assert_eq!(query.original, "test query");
        assert_eq!(query.expanded, "test query");
        assert!(!query.was_expanded);
        assert!(query.expansion_terms.is_empty());
    }

    #[test]
    fn test_expanded_query_with_expansions() {
        let terms = vec![
            ExpansionTerm {
                term: "related".into(),
                score: 0.9,
                source: ExpansionSource::Synonym,
            },
            ExpansionTerm {
                term: "similar".into(),
                score: 0.8,
                source: ExpansionSource::Embedding,
            },
        ];

        let query = ExpandedQuery::with_expansions("test query", terms);
        assert_eq!(query.original, "test query");
        assert_eq!(query.expanded, "test query related similar");
        assert!(query.was_expanded);
        assert_eq!(query.expansion_terms.len(), 2);
    }

    #[test]
    fn test_synonym_expander_default() {
        let expander = SynonymExpander::new();
        assert!(!expander.synonyms.is_empty());
        assert!(expander.synonyms.contains_key("error"));
    }

    #[test]
    fn test_synonym_expander_find_similar() {
        let expander = SynonymExpander::new();
        let similar = expander.find_similar("eror", 2); // typo for "error"
        assert!(similar.contains(&"error".to_string()));
    }

    #[tokio::test]
    async fn test_synonym_expander_expand() {
        let expander = SynonymExpander::new();
        let config = QueryExpansionConfig {
            enabled: true,
            method: crate::config::QueryExpansionMethod::Synonym,
            max_expansions: 3,
            similarity_threshold: 0.7,
            pseudo_relevance: false,
            prf_top_k: 5,
        };

        let result = expander.expand("find error", &config).await.unwrap();
        assert!(result.was_expanded);
        assert!(!result.expansion_terms.is_empty());
    }

    #[test]
    fn test_prf_extract_terms() {
        let prf = PseudoRelevanceFeedbackExpander::new(5);
        let texts = vec![
            "The database connection failed".to_string(),
            "Database timeout error occurred".to_string(),
            "Connection to database refused".to_string(),
        ];

        let terms = prf.extract_terms(&texts, "error");
        assert!(!terms.is_empty());

        // "database" and "connection" should appear in multiple docs
        let term_names: Vec<&str> = terms.iter().map(|t| t.term.as_str()).collect();
        assert!(term_names.contains(&"database"));
        assert!(term_names.contains(&"connection"));
    }
}
