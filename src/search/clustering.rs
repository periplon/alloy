//! Semantic clustering module for document grouping.
//!
//! This module provides clustering capabilities to group similar documents
//! together for exploration and organization.
//!
//! # Supported Algorithms
//!
//! - **K-Means**: Classic centroid-based clustering
//! - **DBSCAN**: Density-based clustering with automatic cluster count

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use linfa::dataset::AsTargets;
use linfa::traits::{Fit, Predict, Transformer};
use linfa::DatasetBase;
use linfa_clustering::{Dbscan, KMeans};
use moka::future::Cache;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::config::{ClusteringAlgorithm, ClusteringConfig};
use crate::embedding::EmbeddingProvider;
use crate::error::Result;

/// A single cluster of documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Unique cluster ID.
    pub cluster_id: usize,
    /// Human-readable label for the cluster.
    pub label: String,
    /// Keywords describing the cluster.
    pub keywords: Vec<String>,
    /// Document IDs in this cluster.
    pub document_ids: Vec<String>,
    /// Number of documents in cluster.
    pub size: usize,
    /// Cluster centroid (average embedding).
    pub centroid: Option<Vec<f32>>,
    /// Coherence score (0.0 to 1.0, higher is more coherent).
    pub coherence_score: f64,
    /// Representative document IDs (closest to centroid).
    pub representative_docs: Vec<String>,
}

/// Result of a clustering operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// The clusters.
    pub clusters: Vec<Cluster>,
    /// Documents that couldn't be assigned to any cluster (outliers).
    pub outliers: Vec<String>,
    /// Quality metrics for the clustering.
    pub metrics: ClusteringMetrics,
    /// Algorithm used.
    pub algorithm_used: String,
    /// Total documents processed.
    pub total_documents: usize,
    /// When the clustering was created.
    pub created_at: DateTime<Utc>,
}

/// Quality metrics for clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringMetrics {
    /// Silhouette score (-1 to 1, higher is better).
    pub silhouette_score: f64,
    /// Inertia/distortion (sum of squared distances to centroids).
    pub inertia: f64,
    /// Calinski-Harabasz index (higher is better, measures between/within variance ratio).
    pub calinski_harabasz_index: f64,
    /// Davies-Bouldin index (lower is better, measures cluster separation).
    pub davies_bouldin_index: f64,
    /// Number of clusters.
    pub num_clusters: usize,
    /// Number of outliers.
    pub num_outliers: usize,
    /// Distribution of cluster sizes.
    pub cluster_size_distribution: Vec<usize>,
}

/// Input for clustering: document ID and its embedding.
#[derive(Debug, Clone)]
pub struct ClusterInput {
    /// Document ID.
    pub document_id: String,
    /// Document embedding vector.
    pub embedding: Vec<f32>,
    /// Optional text for keyword extraction.
    pub text: Option<String>,
}

/// Trait for clustering implementations.
#[async_trait]
pub trait DocumentClusterer: Send + Sync {
    /// Cluster documents based on their embeddings.
    async fn cluster(
        &self,
        inputs: Vec<ClusterInput>,
        config: &ClusteringConfig,
    ) -> Result<ClusteringResult>;

    /// Get the name of this clusterer.
    fn name(&self) -> &str;
}

/// K-Means clustering implementation.
pub struct KMeansClusterer;

impl KMeansClusterer {
    /// Create a new K-Means clusterer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for KMeansClusterer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentClusterer for KMeansClusterer {
    async fn cluster(
        &self,
        inputs: Vec<ClusterInput>,
        config: &ClusteringConfig,
    ) -> Result<ClusteringResult> {
        if inputs.is_empty() {
            return Ok(ClusteringResult {
                clusters: vec![],
                outliers: vec![],
                metrics: ClusteringMetrics {
                    silhouette_score: 0.0,
                    inertia: 0.0,
                    calinski_harabasz_index: 0.0,
                    davies_bouldin_index: 0.0,
                    num_clusters: 0,
                    num_outliers: 0,
                    cluster_size_distribution: vec![],
                },
                algorithm_used: "kmeans".to_string(),
                total_documents: 0,
                created_at: Utc::now(),
            });
        }

        let num_docs = inputs.len();
        let embedding_dim = inputs[0].embedding.len();

        // Convert embeddings to ndarray
        let mut data = Array2::zeros((num_docs, embedding_dim));
        for (i, input) in inputs.iter().enumerate() {
            for (j, &val) in input.embedding.iter().enumerate() {
                data[[i, j]] = val as f64;
            }
        }

        // Determine number of clusters
        let num_clusters = if config.default_num_clusters == 0 {
            // Auto-detect: sqrt(n/2) heuristic
            ((num_docs as f64 / 2.0).sqrt().ceil() as usize)
                .max(2)
                .min(num_docs)
        } else {
            config.default_num_clusters.min(num_docs)
        };

        // Create dataset from records
        let dataset = DatasetBase::from(data.clone());

        // Run K-Means using the proper API
        let model = KMeans::params(num_clusters)
            .max_n_iterations(100)
            .tolerance(1e-4)
            .fit(&dataset)
            .map_err(|e| {
                crate::error::SearchError::Clustering(format!("K-Means fit failed: {:?}", e))
            })?;

        // Get predictions for each point - predict returns an array of cluster assignments
        let predictions = model.predict(&dataset);
        let labels: Vec<usize> = predictions.as_targets().iter().copied().collect();

        // Get centroids
        let centroids = model.centroids();

        // Group documents by cluster
        let mut cluster_docs: HashMap<usize, Vec<(String, Option<String>)>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            cluster_docs
                .entry(*label)
                .or_default()
                .push((inputs[i].document_id.clone(), inputs[i].text.clone()));
        }

        // Build clusters
        let mut clusters = Vec::new();
        let mut cluster_sizes = Vec::new();

        for (cluster_id, docs) in &cluster_docs {
            let centroid: Vec<f32> = centroids
                .index_axis(Axis(0), *cluster_id)
                .iter()
                .map(|&v| v as f32)
                .collect();

            // Extract keywords from document texts
            let keywords = if config.generate_labels {
                extract_keywords(
                    &docs
                        .iter()
                        .filter_map(|(_, t)| t.clone())
                        .collect::<Vec<_>>(),
                    config.max_keywords,
                )
            } else {
                vec![]
            };

            // Generate label
            let label = if keywords.is_empty() {
                format!("Cluster {}", cluster_id)
            } else {
                keywords[..keywords.len().min(3)].join(", ")
            };

            // Find representative documents (closest to centroid)
            let representative_docs = find_representative_docs(
                &docs.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
                3,
            );

            // Compute coherence score
            let coherence = compute_coherence(
                &docs.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
            );

            cluster_sizes.push(docs.len());
            clusters.push(Cluster {
                cluster_id: *cluster_id,
                label,
                keywords,
                document_ids: docs.iter().map(|(id, _)| id.clone()).collect(),
                size: docs.len(),
                centroid: Some(centroid),
                coherence_score: coherence,
                representative_docs,
            });
        }

        // Compute metrics
        let inertia = model.inertia();
        let silhouette = compute_silhouette_score(&data, &labels, num_clusters);
        let calinski_harabasz = compute_calinski_harabasz(&data, &labels, num_clusters);
        let davies_bouldin = compute_davies_bouldin(&data, &labels, num_clusters, centroids);

        Ok(ClusteringResult {
            clusters,
            outliers: vec![], // K-Means doesn't produce outliers
            metrics: ClusteringMetrics {
                silhouette_score: silhouette,
                inertia,
                calinski_harabasz_index: calinski_harabasz,
                davies_bouldin_index: davies_bouldin,
                num_clusters,
                num_outliers: 0,
                cluster_size_distribution: cluster_sizes,
            },
            algorithm_used: "kmeans".to_string(),
            total_documents: num_docs,
            created_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "kmeans"
    }
}

/// DBSCAN clustering implementation (density-based).
pub struct DbscanClusterer {
    /// Epsilon (maximum distance between points in same cluster).
    epsilon: f64,
}

impl DbscanClusterer {
    /// Create a new DBSCAN clusterer.
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for DbscanClusterer {
    fn default() -> Self {
        Self::new(0.5) // Default epsilon
    }
}

#[async_trait]
impl DocumentClusterer for DbscanClusterer {
    async fn cluster(
        &self,
        inputs: Vec<ClusterInput>,
        config: &ClusteringConfig,
    ) -> Result<ClusteringResult> {
        if inputs.is_empty() {
            return Ok(ClusteringResult {
                clusters: vec![],
                outliers: vec![],
                metrics: ClusteringMetrics {
                    silhouette_score: 0.0,
                    inertia: 0.0,
                    calinski_harabasz_index: 0.0,
                    davies_bouldin_index: 0.0,
                    num_clusters: 0,
                    num_outliers: 0,
                    cluster_size_distribution: vec![],
                },
                algorithm_used: "dbscan".to_string(),
                total_documents: 0,
                created_at: Utc::now(),
            });
        }

        let num_docs = inputs.len();
        let embedding_dim = inputs[0].embedding.len();

        // Convert embeddings to ndarray
        let mut data = Array2::zeros((num_docs, embedding_dim));
        for (i, input) in inputs.iter().enumerate() {
            for (j, &val) in input.embedding.iter().enumerate() {
                data[[i, j]] = val as f64;
            }
        }

        // Create DBSCAN model and run clustering
        let min_points = config.min_cluster_size;
        let dataset = DatasetBase::from(data.clone());

        let clusters_result = Dbscan::params(min_points)
            .tolerance(self.epsilon)
            .transform(dataset)
            .map_err(|e| {
                crate::error::SearchError::Clustering(format!("DBSCAN failed: {:?}", e))
            })?;

        // Extract labels (None = outlier)
        let labels: Vec<Option<usize>> = clusters_result.targets().iter().copied().collect();

        // Group documents by cluster
        let mut cluster_docs: HashMap<usize, Vec<(String, Option<String>)>> = HashMap::new();
        let mut outliers: Vec<String> = Vec::new();

        for (i, label) in labels.iter().enumerate() {
            match label {
                Some(cluster_id) => {
                    cluster_docs
                        .entry(*cluster_id)
                        .or_default()
                        .push((inputs[i].document_id.clone(), inputs[i].text.clone()));
                }
                None => {
                    outliers.push(inputs[i].document_id.clone());
                }
            }
        }

        // Build clusters
        let mut clusters = Vec::new();
        let mut cluster_sizes = Vec::new();

        for (cluster_id, docs) in &cluster_docs {
            // Compute centroid for this cluster
            let centroid = compute_centroid(
                &docs.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                &inputs,
            );

            // Extract keywords
            let keywords = if config.generate_labels {
                extract_keywords(
                    &docs
                        .iter()
                        .filter_map(|(_, t)| t.clone())
                        .collect::<Vec<_>>(),
                    config.max_keywords,
                )
            } else {
                vec![]
            };

            let label = if keywords.is_empty() {
                format!("Cluster {}", cluster_id)
            } else {
                keywords[..keywords.len().min(3)].join(", ")
            };

            let representative_docs = find_representative_docs(
                &docs.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
                3,
            );

            let coherence = compute_coherence(
                &docs.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
            );

            cluster_sizes.push(docs.len());
            clusters.push(Cluster {
                cluster_id: *cluster_id,
                label,
                keywords,
                document_ids: docs.iter().map(|(id, _)| id.clone()).collect(),
                size: docs.len(),
                centroid: Some(centroid),
                coherence_score: coherence,
                representative_docs,
            });
        }

        let num_clusters = clusters.len();

        Ok(ClusteringResult {
            clusters,
            outliers: outliers.clone(),
            metrics: ClusteringMetrics {
                silhouette_score: 0.0, // Not computed for DBSCAN with outliers
                inertia: 0.0,
                calinski_harabasz_index: 0.0, // Not computed for DBSCAN
                davies_bouldin_index: 0.0,    // Not computed for DBSCAN
                num_clusters,
                num_outliers: outliers.len(),
                cluster_size_distribution: cluster_sizes,
            },
            algorithm_used: "dbscan".to_string(),
            total_documents: num_docs,
            created_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "dbscan"
    }
}

/// Agglomerative (hierarchical) clustering implementation.
pub struct AgglomerativeClusterer {
    /// Number of clusters (if None, uses distance threshold).
    num_clusters: Option<usize>,
    /// Distance threshold for cutting dendrogram.
    #[allow(dead_code)] // Used for future distance-based clustering
    distance_threshold: Option<f64>,
    /// Linkage type.
    linkage: crate::config::AgglomerativeLinkage,
}

impl AgglomerativeClusterer {
    /// Create a new agglomerative clusterer.
    pub fn new(
        num_clusters: Option<usize>,
        distance_threshold: Option<f64>,
        linkage: crate::config::AgglomerativeLinkage,
    ) -> Self {
        Self {
            num_clusters,
            distance_threshold,
            linkage,
        }
    }
}

impl Default for AgglomerativeClusterer {
    fn default() -> Self {
        Self::new(None, None, crate::config::AgglomerativeLinkage::Ward)
    }
}

#[async_trait]
impl DocumentClusterer for AgglomerativeClusterer {
    async fn cluster(
        &self,
        inputs: Vec<ClusterInput>,
        config: &ClusteringConfig,
    ) -> Result<ClusteringResult> {
        if inputs.is_empty() {
            return Ok(ClusteringResult {
                clusters: vec![],
                outliers: vec![],
                metrics: ClusteringMetrics {
                    silhouette_score: 0.0,
                    inertia: 0.0,
                    calinski_harabasz_index: 0.0,
                    davies_bouldin_index: 0.0,
                    num_clusters: 0,
                    num_outliers: 0,
                    cluster_size_distribution: vec![],
                },
                algorithm_used: "agglomerative".to_string(),
                total_documents: 0,
                created_at: Utc::now(),
            });
        }

        let num_docs = inputs.len();
        let embedding_dim = inputs[0].embedding.len();

        // Convert embeddings to ndarray
        let mut data = Array2::zeros((num_docs, embedding_dim));
        for (i, input) in inputs.iter().enumerate() {
            for (j, &val) in input.embedding.iter().enumerate() {
                data[[i, j]] = val as f64;
            }
        }

        // Determine number of clusters
        let target_clusters = self.num_clusters.unwrap_or_else(|| {
            if config.default_num_clusters == 0 {
                ((num_docs as f64 / 2.0).sqrt().ceil() as usize)
                    .max(2)
                    .min(num_docs)
            } else {
                config.default_num_clusters.min(num_docs)
            }
        });

        // Perform agglomerative clustering using our own implementation
        // since linfa-hierarchical is complex to integrate
        let labels = agglomerative_cluster_impl(&data, target_clusters, &self.linkage);

        // Group documents by cluster
        let mut cluster_docs: HashMap<usize, Vec<(String, Option<String>)>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            cluster_docs
                .entry(*label)
                .or_default()
                .push((inputs[i].document_id.clone(), inputs[i].text.clone()));
        }

        // Build clusters
        let mut clusters = Vec::new();
        let mut cluster_sizes = Vec::new();

        for (cluster_id, docs) in &cluster_docs {
            let centroid = compute_centroid(
                &docs.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                &inputs,
            );

            let keywords = if config.generate_labels {
                extract_keywords(
                    &docs
                        .iter()
                        .filter_map(|(_, t)| t.clone())
                        .collect::<Vec<_>>(),
                    config.max_keywords,
                )
            } else {
                vec![]
            };

            let label = if keywords.is_empty() {
                format!("Cluster {}", cluster_id)
            } else {
                keywords[..keywords.len().min(3)].join(", ")
            };

            let representative_docs = find_representative_docs(
                &docs.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
                3,
            );

            let coherence = compute_coherence(
                &docs.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
                &inputs,
                &centroid,
            );

            cluster_sizes.push(docs.len());
            clusters.push(Cluster {
                cluster_id: *cluster_id,
                label,
                keywords,
                document_ids: docs.iter().map(|(id, _)| id.clone()).collect(),
                size: docs.len(),
                centroid: Some(centroid),
                coherence_score: coherence,
                representative_docs,
            });
        }

        let num_clusters = clusters.len();

        // Compute metrics
        let silhouette = compute_silhouette_score(&data, &labels, num_clusters);
        let calinski_harabasz = compute_calinski_harabasz(&data, &labels, num_clusters);

        Ok(ClusteringResult {
            clusters,
            outliers: vec![], // Agglomerative clustering assigns all points
            metrics: ClusteringMetrics {
                silhouette_score: silhouette,
                inertia: 0.0, // Not applicable for agglomerative
                calinski_harabasz_index: calinski_harabasz,
                davies_bouldin_index: 0.0, // Would need centroids array
                num_clusters,
                num_outliers: 0,
                cluster_size_distribution: cluster_sizes,
            },
            algorithm_used: "agglomerative".to_string(),
            total_documents: num_docs,
            created_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "agglomerative"
    }
}

/// Simple agglomerative clustering implementation using average linkage.
fn agglomerative_cluster_impl(
    data: &Array2<f64>,
    num_clusters: usize,
    linkage: &crate::config::AgglomerativeLinkage,
) -> Vec<usize> {
    let n = data.nrows();

    if n <= num_clusters {
        return (0..n).collect();
    }

    // Compute pairwise distances
    let mut distances = Array2::zeros((n, n));
    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = (0..data.ncols())
                .map(|k| (data[[i, k]] - data[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();
            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }

    // Initialize each point as its own cluster
    let mut cluster_assignments: Vec<usize> = (0..n).collect();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active_clusters: Vec<bool> = vec![true; n];
    let mut current_num_clusters = n;

    // Merge clusters until we reach target
    while current_num_clusters > num_clusters {
        // Find closest pair of clusters
        let mut min_dist = f64::MAX;
        let mut merge_i = 0;
        let mut merge_j = 0;

        // Note: We need indices i and j to pass to cluster_distance and track merge targets
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if !active_clusters[i] {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for j in i + 1..n {
                if !active_clusters[j] {
                    continue;
                }

                let dist = cluster_distance(
                    i,
                    j,
                    &cluster_assignments,
                    &cluster_sizes,
                    &distances,
                    linkage,
                );

                if dist < min_dist {
                    min_dist = dist;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // Merge cluster j into cluster i
        for assignment in cluster_assignments.iter_mut() {
            if *assignment == merge_j {
                *assignment = merge_i;
            }
        }
        cluster_sizes[merge_i] += cluster_sizes[merge_j];
        cluster_sizes[merge_j] = 0;
        active_clusters[merge_j] = false;
        current_num_clusters -= 1;
    }

    // Renumber clusters to be contiguous 0..num_clusters
    let mut cluster_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;

    for assignment in &mut cluster_assignments {
        if let Some(&mapped) = cluster_map.get(assignment) {
            *assignment = mapped;
        } else {
            cluster_map.insert(*assignment, next_id);
            *assignment = next_id;
            next_id += 1;
        }
    }

    cluster_assignments
}

/// Compute distance between two clusters based on linkage type.
fn cluster_distance(
    cluster_i: usize,
    cluster_j: usize,
    assignments: &[usize],
    _sizes: &[usize],
    distances: &Array2<f64>,
    linkage: &crate::config::AgglomerativeLinkage,
) -> f64 {
    let points_i: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == cluster_i)
        .map(|(idx, _)| idx)
        .collect();
    let points_j: Vec<usize> = assignments
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == cluster_j)
        .map(|(idx, _)| idx)
        .collect();

    if points_i.is_empty() || points_j.is_empty() {
        return f64::MAX;
    }

    match linkage {
        crate::config::AgglomerativeLinkage::Single => {
            // Minimum distance between any pair
            let mut min_d = f64::MAX;
            for &i in &points_i {
                for &j in &points_j {
                    min_d = min_d.min(distances[[i, j]]);
                }
            }
            min_d
        }
        crate::config::AgglomerativeLinkage::Complete => {
            // Maximum distance between any pair
            let mut max_d = 0.0f64;
            for &i in &points_i {
                for &j in &points_j {
                    max_d = max_d.max(distances[[i, j]]);
                }
            }
            max_d
        }
        crate::config::AgglomerativeLinkage::Average | crate::config::AgglomerativeLinkage::Ward => {
            // Average distance (Ward approximation using average linkage)
            let mut sum = 0.0;
            let mut count = 0;
            for &i in &points_i {
                for &j in &points_j {
                    sum += distances[[i, j]];
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64
            } else {
                f64::MAX
            }
        }
    }
}

/// Clustering engine with caching support.
pub struct ClusteringEngine {
    embedder: Arc<dyn EmbeddingProvider>,
    cache: Cache<String, ClusteringResult>,
    config: ClusteringConfig,
}

impl ClusteringEngine {
    /// Create a new clustering engine.
    pub fn new(embedder: Arc<dyn EmbeddingProvider>, config: ClusteringConfig) -> Self {
        let cache = Cache::builder()
            .max_capacity(100)
            .time_to_live(Duration::from_secs(config.cache_ttl_secs))
            .build();

        Self {
            embedder,
            cache,
            config,
        }
    }

    /// Cluster documents with optional caching.
    pub async fn cluster_documents(
        &self,
        inputs: Vec<ClusterInput>,
        override_config: Option<&ClusteringConfig>,
    ) -> Result<ClusteringResult> {
        let config = override_config.unwrap_or(&self.config);

        // Generate cache key
        let cache_key = if config.cache_results {
            let mut ids: Vec<&str> = inputs.iter().map(|i| i.document_id.as_str()).collect();
            ids.sort();
            format!("{:?}:{:?}", config.algorithm, ids.join(","))
        } else {
            String::new()
        };

        // Check cache
        if config.cache_results && !cache_key.is_empty() {
            if let Some(cached) = self.cache.get(&cache_key).await {
                return Ok(cached);
            }
        }

        // Create appropriate clusterer
        let clusterer: Box<dyn DocumentClusterer> = match config.algorithm {
            ClusteringAlgorithm::KMeans => Box::new(KMeansClusterer::new()),
            ClusteringAlgorithm::Dbscan => Box::new(DbscanClusterer::new(config.dbscan_epsilon)),
            ClusteringAlgorithm::Gmm => {
                // Fall back to K-Means for now (GMM requires additional implementation)
                Box::new(KMeansClusterer::new())
            }
            ClusteringAlgorithm::Agglomerative => Box::new(AgglomerativeClusterer::new(
                Some(config.default_num_clusters),
                config.agglomerative_distance_threshold,
                config.agglomerative_linkage,
            )),
        };

        // Perform clustering
        let result = clusterer.cluster(inputs, config).await?;

        // Cache result
        if config.cache_results && !cache_key.is_empty() {
            self.cache.insert(cache_key, result.clone()).await;
        }

        Ok(result)
    }

    /// Cluster documents by their text (will compute embeddings).
    pub async fn cluster_texts(
        &self,
        documents: Vec<(String, String)>, // (id, text)
        config: Option<&ClusteringConfig>,
    ) -> Result<ClusteringResult> {
        let texts: Vec<String> = documents.iter().map(|(_, t)| t.clone()).collect();
        let embeddings = self.embedder.embed(&texts).await?;

        let inputs: Vec<ClusterInput> = documents
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|((id, text), embedding)| ClusterInput {
                document_id: id,
                embedding,
                text: Some(text),
            })
            .collect();

        self.cluster_documents(inputs, config).await
    }

    /// Clear the clustering cache.
    pub async fn clear_cache(&self) {
        self.cache.invalidate_all();
    }

    /// Find the most similar cluster for a query.
    pub async fn find_similar_cluster(
        &self,
        query: &str,
        clustering_result: &ClusteringResult,
        top_k: usize,
    ) -> Result<Vec<(usize, f64)>> {
        // Embed the query
        let query_embedding = self.embedder.embed(&[query.to_string()]).await?;
        let query_vec = &query_embedding[0];

        // Compare query embedding to cluster centroids
        let mut similarities: Vec<(usize, f64)> = clustering_result
            .clusters
            .iter()
            .filter_map(|cluster| {
                cluster.centroid.as_ref().map(|centroid| {
                    let sim = cosine_similarity(query_vec, centroid);
                    (cluster.cluster_id, sim)
                })
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        Ok(similarities.into_iter().take(top_k).collect())
    }

    /// Re-generate labels for clusters using the configured labeling method.
    pub async fn regenerate_labels(
        &self,
        result: &mut ClusteringResult,
        labeler: &dyn ClusterLabeler,
    ) -> Result<()> {
        for cluster in &mut result.clusters {
            let texts: Vec<&str> = cluster.document_ids.iter().map(|s| s.as_str()).collect();
            if let Ok(label) = labeler.generate_label(&texts, &cluster.keywords).await {
                cluster.label = label;
            }
        }
        Ok(())
    }
}

/// Trait for cluster label generation.
#[async_trait]
pub trait ClusterLabeler: Send + Sync {
    /// Generate a descriptive label for a cluster.
    async fn generate_label(&self, sample_texts: &[&str], keywords: &[String]) -> Result<String>;

    /// Get the name of this labeler.
    fn name(&self) -> &str;
}

/// Keyword-based cluster labeler (TF-IDF).
pub struct KeywordLabeler {
    max_keywords: usize,
}

impl KeywordLabeler {
    /// Create a new keyword labeler.
    pub fn new(max_keywords: usize) -> Self {
        Self { max_keywords }
    }
}

impl Default for KeywordLabeler {
    fn default() -> Self {
        Self::new(5)
    }
}

#[async_trait]
impl ClusterLabeler for KeywordLabeler {
    async fn generate_label(&self, _sample_texts: &[&str], keywords: &[String]) -> Result<String> {
        if keywords.is_empty() {
            Ok("Unlabeled Cluster".to_string())
        } else {
            let label_keywords: Vec<&str> = keywords
                .iter()
                .take(self.max_keywords.min(3))
                .map(|s| s.as_str())
                .collect();
            Ok(label_keywords.join(", "))
        }
    }

    fn name(&self) -> &str {
        "keyword"
    }
}

/// LLM-based cluster labeler.
pub struct LlmLabeler {
    api_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl LlmLabeler {
    /// Create a new LLM labeler.
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create from configuration.
    pub fn from_config(config: &crate::config::ClusterLabelingConfig) -> Option<Self> {
        let api_url = config.llm_api_url.as_deref()
            .unwrap_or("https://api.openai.com/v1");

        // Try config API key first, then environment variable
        let api_key = config.llm_api_key.clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())?;

        Some(Self::new(api_url, &api_key, &config.llm_model))
    }
}

#[async_trait]
impl ClusterLabeler for LlmLabeler {
    async fn generate_label(&self, sample_texts: &[&str], keywords: &[String]) -> Result<String> {
        let sample_text = sample_texts
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(
            "Generate a short (3-5 word) descriptive label for a cluster of documents.\n\n\
            Keywords extracted from the cluster: {}\n\n\
            Sample documents:\n{}\n\n\
            Label (3-5 words only):",
            keywords.join(", "),
            sample_text
        );

        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise, descriptive labels for document clusters. Respond with only the label, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 20,
            "temperature": 0.3
        });

        let response = self.client
            .post(format!("{}/chat/completions", self.api_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| crate::error::SearchError::Clustering(format!("LLM API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::SearchError::Clustering(
                format!("LLM API error {}: {}", status, body)
            ).into());
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| crate::error::SearchError::Clustering(format!("Failed to parse LLM response: {}", e)))?;

        let label = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("Unlabeled Cluster")
            .trim()
            .to_string();

        Ok(label)
    }

    fn name(&self) -> &str {
        "llm"
    }
}

/// Hybrid labeler that combines keywords with LLM.
pub struct HybridLabeler {
    keyword_labeler: KeywordLabeler,
    llm_labeler: Option<LlmLabeler>,
}

impl HybridLabeler {
    /// Create a new hybrid labeler.
    pub fn new(max_keywords: usize, llm_labeler: Option<LlmLabeler>) -> Self {
        Self {
            keyword_labeler: KeywordLabeler::new(max_keywords),
            llm_labeler,
        }
    }
}

#[async_trait]
impl ClusterLabeler for HybridLabeler {
    async fn generate_label(&self, sample_texts: &[&str], keywords: &[String]) -> Result<String> {
        // Try LLM first if available
        if let Some(ref llm) = self.llm_labeler {
            match llm.generate_label(sample_texts, keywords).await {
                Ok(label) => return Ok(label),
                Err(e) => {
                    tracing::warn!("LLM labeling failed, falling back to keywords: {}", e);
                }
            }
        }

        // Fall back to keyword labeling
        self.keyword_labeler.generate_label(sample_texts, keywords).await
    }

    fn name(&self) -> &str {
        "hybrid"
    }
}

/// Factory for creating labelers based on configuration.
pub struct LabelerFactory;

impl LabelerFactory {
    /// Create a labeler based on configuration.
    pub fn create(config: &crate::config::ClusterLabelingConfig) -> Box<dyn ClusterLabeler> {
        match config.method {
            crate::config::ClusterLabelingMethod::Keywords => {
                Box::new(KeywordLabeler::new(config.max_keywords))
            }
            crate::config::ClusterLabelingMethod::Llm => {
                if let Some(labeler) = LlmLabeler::from_config(config) {
                    Box::new(labeler)
                } else {
                    tracing::warn!("LLM labeler not configured, falling back to keywords");
                    Box::new(KeywordLabeler::new(config.max_keywords))
                }
            }
            crate::config::ClusterLabelingMethod::Hybrid => {
                let llm_labeler = LlmLabeler::from_config(config);
                Box::new(HybridLabeler::new(config.max_keywords, llm_labeler))
            }
        }
    }
}

// Helper functions

/// Extract keywords from a collection of texts using TF-IDF-like scoring.
fn extract_keywords(texts: &[String], max_keywords: usize) -> Vec<String> {
    use std::collections::HashSet;

    if texts.is_empty() {
        return vec![];
    }

    // Common English stopwords
    let stopwords: HashSet<&str> = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can",
        "this", "that", "these", "those", "it", "its", "as", "if", "then", "else", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "also",
    ]
    .iter()
    .copied()
    .collect();

    // Count term frequencies
    let mut term_freq: HashMap<String, usize> = HashMap::new();
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for text in texts {
        let mut seen_in_doc: HashSet<String> = HashSet::new();

        for word in text.split_whitespace() {
            let cleaned = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();

            if cleaned.len() >= 3 && !stopwords.contains(cleaned.as_str()) {
                *term_freq.entry(cleaned.clone()).or_insert(0) += 1;

                if !seen_in_doc.contains(&cleaned) {
                    seen_in_doc.insert(cleaned.clone());
                    *doc_freq.entry(cleaned).or_insert(0) += 1;
                }
            }
        }
    }

    // Calculate TF-IDF scores
    let num_docs = texts.len() as f64;
    let mut scores: Vec<(String, f64)> = term_freq
        .into_iter()
        .map(|(term, tf)| {
            let df = doc_freq.get(&term).copied().unwrap_or(1) as f64;
            let idf = (num_docs / df).ln() + 1.0;
            let score = (tf as f64) * idf;
            (term, score)
        })
        .collect();

    // Sort by score and take top keywords
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
        .into_iter()
        .take(max_keywords)
        .map(|(term, _)| term)
        .collect()
}

/// Compute centroid (average embedding) for a set of documents.
fn compute_centroid(doc_ids: &[&str], inputs: &[ClusterInput]) -> Vec<f32> {
    let doc_set: std::collections::HashSet<&str> = doc_ids.iter().copied().collect();
    let relevant: Vec<&ClusterInput> = inputs
        .iter()
        .filter(|i| doc_set.contains(i.document_id.as_str()))
        .collect();

    if relevant.is_empty() {
        return vec![];
    }

    let dim = relevant[0].embedding.len();
    let mut centroid = vec![0.0f32; dim];

    for input in &relevant {
        for (i, &val) in input.embedding.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let count = relevant.len() as f32;
    for val in &mut centroid {
        *val /= count;
    }

    centroid
}

/// Find documents closest to a centroid.
fn find_representative_docs(
    doc_ids: &[String],
    inputs: &[ClusterInput],
    centroid: &[f32],
    max_docs: usize,
) -> Vec<String> {
    let doc_set: std::collections::HashSet<&str> = doc_ids.iter().map(|s| s.as_str()).collect();

    let mut scored: Vec<(&String, f64)> = inputs
        .iter()
        .filter(|i| doc_set.contains(i.document_id.as_str()))
        .map(|i| {
            let dist = euclidean_distance(&i.embedding, centroid);
            (&i.document_id, dist)
        })
        .collect();

    // Sort by distance (ascending = closest first)
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(max_docs)
        .map(|(id, _)| id.clone())
        .collect()
}

/// Compute coherence score (average similarity to centroid).
fn compute_coherence(doc_ids: &[&str], inputs: &[ClusterInput], centroid: &[f32]) -> f64 {
    let doc_set: std::collections::HashSet<&str> = doc_ids.iter().copied().collect();

    let similarities: Vec<f64> = inputs
        .iter()
        .filter(|i| doc_set.contains(i.document_id.as_str()))
        .map(|i| cosine_similarity(&i.embedding, centroid))
        .collect();

    if similarities.is_empty() {
        return 0.0;
    }

    similarities.iter().sum::<f64>() / similarities.len() as f64
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute simplified silhouette score.
fn compute_silhouette_score(data: &Array2<f64>, labels: &[usize], num_clusters: usize) -> f64 {
    if num_clusters <= 1 || data.nrows() <= num_clusters {
        return 0.0;
    }

    let n = data.nrows();
    let mut silhouettes = Vec::with_capacity(n);

    for i in 0..n {
        let point = data.row(i);
        let label = labels[i];

        // Compute average distance to same-cluster points
        let mut same_cluster_sum = 0.0;
        let mut same_cluster_count = 0;

        for (j, &other_label) in labels.iter().enumerate() {
            if i != j && other_label == label {
                let other = data.row(j);
                let dist: f64 = point
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                same_cluster_sum += dist;
                same_cluster_count += 1;
            }
        }

        let a = if same_cluster_count > 0 {
            same_cluster_sum / same_cluster_count as f64
        } else {
            0.0
        };

        // Compute minimum average distance to other clusters
        let mut min_b = f64::MAX;

        for other_cluster in 0..num_clusters {
            if other_cluster == label {
                continue;
            }

            let mut other_sum = 0.0;
            let mut other_count = 0;

            for (j, &other_label) in labels.iter().enumerate() {
                if other_label == other_cluster {
                    let other = data.row(j);
                    let dist: f64 = point
                        .iter()
                        .zip(other.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    other_sum += dist;
                    other_count += 1;
                }
            }

            if other_count > 0 {
                let avg = other_sum / other_count as f64;
                min_b = min_b.min(avg);
            }
        }

        let b = if min_b == f64::MAX { 0.0 } else { min_b };

        let silhouette = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };

        silhouettes.push(silhouette);
    }

    if silhouettes.is_empty() {
        0.0
    } else {
        silhouettes.iter().sum::<f64>() / silhouettes.len() as f64
    }
}

/// Compute Calinski-Harabasz index (Variance Ratio Criterion).
///
/// Higher values indicate better-defined clusters.
/// It measures the ratio of between-cluster dispersion to within-cluster dispersion.
fn compute_calinski_harabasz(data: &Array2<f64>, labels: &[usize], num_clusters: usize) -> f64 {
    if num_clusters <= 1 || data.nrows() <= num_clusters {
        return 0.0;
    }

    let n = data.nrows();
    let d = data.ncols();

    // Compute overall centroid
    let overall_centroid: Vec<f64> = (0..d)
        .map(|j| data.column(j).mean().unwrap_or(0.0))
        .collect();

    // Compute cluster centroids and sizes
    let mut cluster_centroids: Vec<Vec<f64>> = vec![vec![0.0; d]; num_clusters];
    let mut cluster_sizes: Vec<usize> = vec![0; num_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label < num_clusters {
            cluster_sizes[label] += 1;
            for j in 0..d {
                cluster_centroids[label][j] += data[[i, j]];
            }
        }
    }

    for k in 0..num_clusters {
        if cluster_sizes[k] > 0 {
            for val in cluster_centroids[k].iter_mut() {
                *val /= cluster_sizes[k] as f64;
            }
        }
    }

    // Compute between-cluster dispersion (B)
    let mut between_var = 0.0;
    for k in 0..num_clusters {
        let n_k = cluster_sizes[k] as f64;
        if n_k > 0.0 {
            let dist_sq: f64 = cluster_centroids[k]
                .iter()
                .zip(overall_centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            between_var += n_k * dist_sq;
        }
    }

    // Compute within-cluster dispersion (W)
    let mut within_var = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        if label < num_clusters {
            let dist_sq: f64 = (0..d)
                .map(|j| (data[[i, j]] - cluster_centroids[label][j]).powi(2))
                .sum();
            within_var += dist_sq;
        }
    }

    if within_var == 0.0 {
        return 0.0;
    }

    let k = num_clusters as f64;
    let n_f = n as f64;

    // CH = (B / (k-1)) / (W / (n-k))
    (between_var / (k - 1.0)) / (within_var / (n_f - k))
}

/// Compute Davies-Bouldin index.
///
/// Lower values indicate better-defined clusters.
/// It measures the average similarity between each cluster and its most similar cluster.
fn compute_davies_bouldin(
    data: &Array2<f64>,
    labels: &[usize],
    num_clusters: usize,
    centroids: &Array2<f64>,
) -> f64 {
    if num_clusters <= 1 || data.nrows() <= num_clusters {
        return 0.0;
    }

    let d = data.ncols();

    // Compute intra-cluster dispersion (average distance from centroid)
    let mut dispersions: Vec<f64> = vec![0.0; num_clusters];
    let mut cluster_sizes: Vec<usize> = vec![0; num_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label < num_clusters {
            let dist: f64 = (0..d)
                .map(|j| (data[[i, j]] - centroids[[label, j]]).powi(2))
                .sum::<f64>()
                .sqrt();
            dispersions[label] += dist;
            cluster_sizes[label] += 1;
        }
    }

    // Average dispersion for each cluster
    for k in 0..num_clusters {
        if cluster_sizes[k] > 0 {
            dispersions[k] /= cluster_sizes[k] as f64;
        }
    }

    // Compute Davies-Bouldin index
    let mut db_sum = 0.0;
    for i in 0..num_clusters {
        if cluster_sizes[i] == 0 {
            continue;
        }

        let mut max_ratio: f64 = 0.0;
        for j in 0..num_clusters {
            if i == j || cluster_sizes[j] == 0 {
                continue;
            }

            // Distance between cluster centroids
            let centroid_dist: f64 = (0..d)
                .map(|k| (centroids[[i, k]] - centroids[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();

            if centroid_dist > 0.0 {
                let ratio = (dispersions[i] + dispersions[j]) / centroid_dist;
                max_ratio = max_ratio.max(ratio);
            }
        }
        db_sum += max_ratio;
    }

    let valid_clusters = cluster_sizes.iter().filter(|&&s| s > 0).count();
    if valid_clusters > 0 {
        db_sum / valid_clusters as f64
    } else {
        0.0
    }
}

/// Factory for creating clusterers based on configuration.
pub struct ClustererFactory;

impl ClustererFactory {
    /// Create a clusterer based on configuration.
    pub fn create(config: &ClusteringConfig) -> Box<dyn DocumentClusterer> {
        match config.algorithm {
            ClusteringAlgorithm::KMeans => Box::new(KMeansClusterer::new()),
            ClusteringAlgorithm::Dbscan => Box::new(DbscanClusterer::new(config.dbscan_epsilon)),
            ClusteringAlgorithm::Gmm => Box::new(KMeansClusterer::new()), // Fallback
            ClusteringAlgorithm::Agglomerative => Box::new(AgglomerativeClusterer::new(
                Some(config.default_num_clusters),
                config.agglomerative_distance_threshold,
                config.agglomerative_linkage,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords() {
        let texts = vec![
            "The database connection failed with timeout error".to_string(),
            "Database query returned error for invalid connection".to_string(),
            "Connection timeout during database operation".to_string(),
        ];

        let keywords = extract_keywords(&texts, 3);
        assert!(!keywords.is_empty());

        // "database" and "connection" should appear
        let keyword_set: std::collections::HashSet<_> = keywords.iter().collect();
        assert!(
            keyword_set.contains(&"database".to_string())
                || keyword_set.contains(&"connection".to_string())
        );
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_compute_centroid() {
        let inputs = vec![
            ClusterInput {
                document_id: "doc1".to_string(),
                embedding: vec![1.0, 2.0, 3.0],
                text: None,
            },
            ClusterInput {
                document_id: "doc2".to_string(),
                embedding: vec![3.0, 4.0, 5.0],
                text: None,
            },
        ];

        let centroid = compute_centroid(&["doc1", "doc2"], &inputs);
        assert_eq!(centroid, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cluster_metrics() {
        let metrics = ClusteringMetrics {
            silhouette_score: 0.75,
            inertia: 100.0,
            calinski_harabasz_index: 150.5,
            davies_bouldin_index: 0.5,
            num_clusters: 3,
            num_outliers: 2,
            cluster_size_distribution: vec![10, 15, 8],
        };

        assert_eq!(metrics.num_clusters, 3);
        assert_eq!(metrics.cluster_size_distribution.iter().sum::<usize>(), 33);
        assert!(metrics.calinski_harabasz_index > 0.0);
        assert!(metrics.davies_bouldin_index > 0.0);
    }
}
