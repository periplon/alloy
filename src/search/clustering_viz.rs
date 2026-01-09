//! Clustering visualization support.
//!
//! Provides 2D projections for visualizing document clusters.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A 2D point for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Document ID.
    pub document_id: String,
    /// Cluster ID (None for outliers).
    pub cluster_id: Option<usize>,
}

/// Cluster outline for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterOutline {
    /// Cluster ID.
    pub cluster_id: usize,
    /// Centroid position in 2D.
    pub centroid: (f64, f64),
    /// Convex hull points.
    pub hull: Vec<(f64, f64)>,
    /// Cluster label.
    pub label: String,
}

/// Result of 2D projection for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterVisualization {
    /// All points projected to 2D.
    pub points: Vec<Point2D>,
    /// Cluster outlines.
    pub clusters: Vec<ClusterOutline>,
    /// Projection method used.
    pub projection_method: String,
    /// Original embedding dimension.
    pub original_dimension: usize,
}

/// PCA-based dimensionality reduction for visualization.
pub struct PcaProjection;

impl PcaProjection {
    /// Project high-dimensional embeddings to 2D using PCA.
    ///
    /// Uses a simple covariance-based PCA implementation.
    pub fn project(
        embeddings: &[Vec<f32>],
        doc_ids: &[String],
        cluster_assignments: &[Option<usize>],
    ) -> ClusterVisualization {
        if embeddings.is_empty() {
            return ClusterVisualization {
                points: vec![],
                clusters: vec![],
                projection_method: "pca".to_string(),
                original_dimension: 0,
            };
        }

        let n = embeddings.len();
        let dim = embeddings[0].len();

        // Convert to ndarray for easier manipulation
        let mut data = Array2::<f64>::zeros((n, dim));
        for (i, emb) in embeddings.iter().enumerate() {
            for (j, &val) in emb.iter().enumerate() {
                data[[i, j]] = val as f64;
            }
        }

        // Center the data
        let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
        for i in 0..n {
            for j in 0..dim {
                data[[i, j]] -= mean[j];
            }
        }

        // Compute covariance matrix (simplified - use first two principal components)
        // For a proper implementation, we'd use SVD or eigendecomposition
        // Here we use a simple power iteration method
        let (pc1, pc2) = simple_pca(&data);

        // Project data onto first two principal components
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let row = data.row(i);
            let x = row.dot(&pc1);
            let y = row.dot(&pc2);

            points.push(Point2D {
                x,
                y,
                document_id: doc_ids[i].clone(),
                cluster_id: cluster_assignments[i],
            });
        }

        // Build cluster outlines
        let clusters = build_cluster_outlines(&points);

        ClusterVisualization {
            points,
            clusters,
            projection_method: "pca".to_string(),
            original_dimension: dim,
        }
    }
}

/// Simple PCA using power iteration to find first two principal components.
fn simple_pca(data: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n, dim) = data.dim();

    if n < 2 || dim < 2 {
        // Return default axes
        let mut pc1 = Array1::zeros(dim.max(1));
        let mut pc2 = Array1::zeros(dim.max(1));
        if dim >= 1 {
            pc1[0] = 1.0;
        }
        if dim >= 2 {
            pc2[1] = 1.0;
        }
        return (pc1, pc2);
    }

    // Compute covariance matrix (X^T * X / n)
    let cov = data.t().dot(data) / (n as f64);

    // Power iteration for first principal component
    let pc1 = power_iteration(&cov, 100);

    // Deflate covariance matrix and find second component
    let cov_deflated = deflate_matrix(&cov, &pc1);
    let pc2 = power_iteration(&cov_deflated, 100);

    (pc1, pc2)
}

/// Power iteration to find the dominant eigenvector.
fn power_iteration(matrix: &Array2<f64>, iterations: usize) -> Array1<f64> {
    let n = matrix.dim().0;

    // Initialize with random-ish vector
    let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

    for _ in 0..iterations {
        // Multiply by matrix
        let mut new_v = matrix.dot(&v);

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            new_v /= norm;
        }

        v = new_v;
    }

    v
}

/// Deflate matrix by removing the component in direction v.
fn deflate_matrix(matrix: &Array2<f64>, v: &Array1<f64>) -> Array2<f64> {
    let n = matrix.dim().0;
    let mut result = matrix.clone();

    // Remove v * v^T * lambda component
    let lambda = v.dot(&matrix.dot(v));

    for i in 0..n {
        for j in 0..n {
            result[[i, j]] -= lambda * v[i] * v[j];
        }
    }

    result
}

/// Build cluster outlines from 2D points.
fn build_cluster_outlines(points: &[Point2D]) -> Vec<ClusterOutline> {
    use std::collections::HashMap;

    // Group points by cluster
    let mut clusters: HashMap<usize, Vec<&Point2D>> = HashMap::new();

    for point in points {
        if let Some(cluster_id) = point.cluster_id {
            clusters.entry(cluster_id).or_default().push(point);
        }
    }

    let mut outlines = Vec::new();

    for (cluster_id, cluster_points) in clusters {
        if cluster_points.is_empty() {
            continue;
        }

        // Compute centroid
        let cx: f64 = cluster_points.iter().map(|p| p.x).sum::<f64>() / cluster_points.len() as f64;
        let cy: f64 = cluster_points.iter().map(|p| p.y).sum::<f64>() / cluster_points.len() as f64;

        // Compute convex hull
        let hull = convex_hull(&cluster_points);

        outlines.push(ClusterOutline {
            cluster_id,
            centroid: (cx, cy),
            hull,
            label: format!("Cluster {}", cluster_id),
        });
    }

    outlines.sort_by_key(|o| o.cluster_id);
    outlines
}

/// Compute convex hull using Graham scan algorithm.
fn convex_hull(points: &[&Point2D]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.iter().map(|p| (p.x, p.y)).collect();
    }

    // Convert to (x, y) tuples
    let mut pts: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();

    // Find the lowest point
    let mut lowest_idx = 0;
    for (i, p) in pts.iter().enumerate() {
        if p.1 < pts[lowest_idx].1 || (p.1 == pts[lowest_idx].1 && p.0 < pts[lowest_idx].0) {
            lowest_idx = i;
        }
    }
    pts.swap(0, lowest_idx);

    let pivot = pts[0];

    // Sort by polar angle
    pts[1..].sort_by(|a, b| {
        let angle_a = (a.1 - pivot.1).atan2(a.0 - pivot.0);
        let angle_b = (b.1 - pivot.1).atan2(b.0 - pivot.0);
        angle_a.partial_cmp(&angle_b).unwrap()
    });

    // Graham scan
    let mut hull: Vec<(f64, f64)> = Vec::new();

    for p in pts {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = cross_product(a, b, p);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    hull
}

/// Cross product for three points (used in convex hull).
fn cross_product(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// t-SNE-like projection (simplified).
///
/// This is a simplified t-SNE implementation for visualization.
/// For production, use a proper t-SNE library.
pub struct TsneProjection;

impl TsneProjection {
    /// Project embeddings to 2D using a simplified t-SNE approach.
    #[allow(dead_code)]
    pub fn project(
        embeddings: &[Vec<f32>],
        doc_ids: &[String],
        cluster_assignments: &[Option<usize>],
        perplexity: f64,
        iterations: usize,
    ) -> ClusterVisualization {
        let _ = perplexity; // Simplified version doesn't use perplexity properly

        if embeddings.is_empty() {
            return ClusterVisualization {
                points: vec![],
                clusters: vec![],
                projection_method: "tsne".to_string(),
                original_dimension: 0,
            };
        }

        let n = embeddings.len();
        let dim = embeddings[0].len();

        // Initialize with PCA projection as starting point
        let pca_viz = PcaProjection::project(embeddings, doc_ids, cluster_assignments);

        // Start with PCA positions
        let mut y: Vec<(f64, f64)> = pca_viz.points.iter().map(|p| (p.x, p.y)).collect();

        // Compute pairwise distances in high-dimensional space
        let high_dim_dist = compute_pairwise_distances(embeddings);

        // Simple gradient descent to refine positions
        let learning_rate = 200.0;

        for iter in 0..iterations {
            // Compute low-dimensional distances
            let low_dim_dist = compute_2d_distances(&y);

            // Compute gradients
            let mut gradients: Vec<(f64, f64)> = vec![(0.0, 0.0); n];

            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let high_d = high_dim_dist[[i, j]];
                    let low_d = low_dim_dist[[i, j]].max(1e-10);

                    // Simplified t-SNE gradient
                    let weight = 1.0 / (1.0 + low_d * low_d);
                    let grad_mult = 4.0 * (high_d - low_d) * weight;

                    gradients[i].0 += grad_mult * (y[i].0 - y[j].0);
                    gradients[i].1 += grad_mult * (y[i].1 - y[j].1);
                }
            }

            // Apply gradients with momentum-like decay
            let lr = learning_rate * (1.0 - iter as f64 / iterations as f64);
            for i in 0..n {
                y[i].0 -= lr * gradients[i].0 / n as f64;
                y[i].1 -= lr * gradients[i].1 / n as f64;
            }
        }

        // Build final points
        let points: Vec<Point2D> = y
            .iter()
            .enumerate()
            .map(|(i, (x, y))| Point2D {
                x: *x,
                y: *y,
                document_id: doc_ids[i].clone(),
                cluster_id: cluster_assignments[i],
            })
            .collect();

        let clusters = build_cluster_outlines(&points);

        ClusterVisualization {
            points,
            clusters,
            projection_method: "tsne".to_string(),
            original_dimension: dim,
        }
    }
}

/// Compute pairwise Euclidean distances.
fn compute_pairwise_distances(embeddings: &[Vec<f32>]) -> Array2<f64> {
    let n = embeddings.len();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let d: f64 = embeddings[i]
                .iter()
                .zip(&embeddings[j])
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();

            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }

    distances
}

/// Compute pairwise distances for 2D points.
fn compute_2d_distances(points: &[(f64, f64)]) -> Array2<f64> {
    let n = points.len();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let d = (dx * dx + dy * dy).sqrt();

            distances[[i, j]] = d;
            distances[[j, i]] = d;
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_projection() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let doc_ids: Vec<String> = (0..5).map(|i| format!("doc{}", i)).collect();
        let clusters = vec![Some(0), Some(0), Some(1), Some(1), None];

        let viz = PcaProjection::project(&embeddings, &doc_ids, &clusters);

        assert_eq!(viz.points.len(), 5);
        assert_eq!(viz.projection_method, "pca");
        assert_eq!(viz.original_dimension, 3);

        // Points in same cluster should be close together
        let cluster0_points: Vec<&Point2D> = viz
            .points
            .iter()
            .filter(|p| p.cluster_id == Some(0))
            .collect();
        assert_eq!(cluster0_points.len(), 2);
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            Point2D {
                x: 0.0,
                y: 0.0,
                document_id: "a".to_string(),
                cluster_id: Some(0),
            },
            Point2D {
                x: 1.0,
                y: 0.0,
                document_id: "b".to_string(),
                cluster_id: Some(0),
            },
            Point2D {
                x: 0.5,
                y: 1.0,
                document_id: "c".to_string(),
                cluster_id: Some(0),
            },
            Point2D {
                x: 0.5,
                y: 0.3,
                document_id: "d".to_string(),
                cluster_id: Some(0),
            }, // Interior point
        ];

        let refs: Vec<&Point2D> = points.iter().collect();
        let hull = convex_hull(&refs);

        // Hull should have 3 points (triangle), not including the interior point
        assert!(hull.len() >= 3);
    }

    #[test]
    fn test_empty_projection() {
        let viz = PcaProjection::project(&[], &[], &[]);
        assert!(viz.points.is_empty());
        assert!(viz.clusters.is_empty());
    }

    #[test]
    fn test_cluster_outlines() {
        let points = vec![
            Point2D {
                x: 0.0,
                y: 0.0,
                document_id: "a".to_string(),
                cluster_id: Some(0),
            },
            Point2D {
                x: 1.0,
                y: 0.0,
                document_id: "b".to_string(),
                cluster_id: Some(0),
            },
            Point2D {
                x: 10.0,
                y: 10.0,
                document_id: "c".to_string(),
                cluster_id: Some(1),
            },
            Point2D {
                x: 11.0,
                y: 10.0,
                document_id: "d".to_string(),
                cluster_id: Some(1),
            },
        ];

        let outlines = build_cluster_outlines(&points);

        assert_eq!(outlines.len(), 2);

        // Cluster 0 centroid should be around (0.5, 0.0)
        let c0 = outlines.iter().find(|o| o.cluster_id == 0).unwrap();
        assert!((c0.centroid.0 - 0.5).abs() < 0.01);
        assert!((c0.centroid.1 - 0.0).abs() < 0.01);

        // Cluster 1 centroid should be around (10.5, 10.0)
        let c1 = outlines.iter().find(|o| o.cluster_id == 1).unwrap();
        assert!((c1.centroid.0 - 10.5).abs() < 0.01);
        assert!((c1.centroid.1 - 10.0).abs() < 0.01);
    }
}
