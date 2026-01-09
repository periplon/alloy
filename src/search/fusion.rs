//! Fusion algorithms for hybrid search.

use serde::{Deserialize, Serialize};

/// Fusion algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FusionAlgorithm {
    /// Reciprocal Rank Fusion
    #[default]
    Rrf,
    /// Distribution-Based Score Fusion
    Dbsf,
}

/// Reciprocal Rank Fusion (RRF) implementation.
///
/// RRF(d) = Σ 1 / (k + rank_i(d))
/// where k is a constant (typically 60) and rank_i(d) is the rank of document d in result list i.
pub fn rrf_fusion(results: &[Vec<(String, f32)>], k: f32, weights: &[f32]) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    if results.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (list_idx, result_list) in results.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);

        for (rank, (doc_id, _score)) in result_list.iter().enumerate() {
            let rrf_score = weight / (k + (rank + 1) as f32);
            *scores.entry(doc_id.clone()).or_insert(0.0) += rrf_score;
        }
    }

    let mut sorted: Vec<_> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Distribution-Based Score Fusion (DBSF) implementation.
///
/// Normalizes scores based on distribution and combines them.
pub fn dbsf_fusion(results: &[Vec<(String, f32)>], weights: &[f32]) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    if results.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (list_idx, result_list) in results.iter().enumerate() {
        if result_list.is_empty() {
            continue;
        }

        let weight = weights.get(list_idx).copied().unwrap_or(1.0);

        // Calculate mean and std dev for normalization
        let raw_scores: Vec<f32> = result_list.iter().map(|(_, s)| *s).collect();
        let mean = raw_scores.iter().sum::<f32>() / raw_scores.len() as f32;
        let variance =
            raw_scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / raw_scores.len() as f32;
        let std_dev = variance.sqrt().max(0.001); // Avoid division by zero

        for (doc_id, score) in result_list {
            // Z-score normalization
            let normalized = (score - mean) / std_dev;
            // Scale to 0-1 range using sigmoid-like function
            let scaled = 1.0 / (1.0 + (-normalized).exp());
            *scores.entry(doc_id.clone()).or_insert(0.0) += scaled * weight;
        }
    }

    let mut sorted: Vec<_> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let results = vec![
            vec![
                ("doc1".to_string(), 0.9),
                ("doc2".to_string(), 0.8),
                ("doc3".to_string(), 0.7),
            ],
            vec![
                ("doc2".to_string(), 0.95),
                ("doc1".to_string(), 0.85),
                ("doc4".to_string(), 0.75),
            ],
        ];

        let fused = rrf_fusion(&results, 60.0, &[1.0, 1.0]);

        // doc1 and doc2 should be at the top since they appear in both lists
        assert!(!fused.is_empty());
        let top_docs: Vec<_> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_docs.contains(&"doc1"));
        assert!(top_docs.contains(&"doc2"));
    }

    #[test]
    fn test_dbsf_fusion() {
        let results = vec![
            vec![("doc1".to_string(), 0.9), ("doc2".to_string(), 0.5)],
            vec![("doc2".to_string(), 0.8), ("doc3".to_string(), 0.3)],
        ];

        let fused = dbsf_fusion(&results, &[0.5, 0.5]);

        assert!(!fused.is_empty());
        // doc2 appears in both lists with decent scores
        assert_eq!(fused[0].0, "doc2");
    }

    #[test]
    fn test_empty_results() {
        let results: Vec<Vec<(String, f32)>> = vec![];
        assert!(rrf_fusion(&results, 60.0, &[]).is_empty());
        assert!(dbsf_fusion(&results, &[]).is_empty());
    }
}
