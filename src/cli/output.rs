//! Output formatting for CLI commands.
//!
//! This module handles formatting output as either JSON or human-readable text.

use alloy::mcp::{
    DocumentDetails, IndexPathResponse, IndexStats, ListSourcesResponse, RemoveSourceResponse,
    SearchResponse,
};

/// Print index result.
pub fn print_index_result(result: &IndexPathResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        println!("Indexed {} documents", result.documents_indexed);
        println!("Source ID: {}", result.source_id);
        println!("Chunks created: {}", result.chunks_created);
        println!("Watching: {}", result.watching);
        if !result.message.is_empty() {
            println!("{}", result.message);
        }
    }
}

/// Print search results.
pub fn print_search_results(result: &SearchResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        println!(
            "Found {} results ({}ms)\n",
            result.total_matches, result.took_ms
        );

        for (i, r) in result.results.iter().enumerate() {
            println!("{}. [{:.3}] {}", i + 1, r.score, r.path);
            let preview = if r.content.len() > 100 {
                format!("{}...", &r.content[..100].replace('\n', " "))
            } else {
                r.content.replace('\n', " ")
            };
            println!("   \"{}\"\n", preview);
        }

        if result.results.is_empty() {
            println!("No results found.");
        }
    }
}

/// Print document details.
pub fn print_document(result: &Option<DocumentDetails>, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if let Some(doc) = result {
        println!("Document ID: {}", doc.document_id);
        println!("Source ID: {}", doc.source_id);
        println!("Path: {}", doc.path);
        println!("MIME Type: {}", doc.mime_type);
        println!("Size: {} bytes", doc.size_bytes);
        println!("Chunks: {}", doc.chunk_count);
        println!("Modified: {}", doc.modified_at);
        println!("Indexed: {}", doc.indexed_at);

        if let Some(content) = &doc.content {
            println!("\n--- Content ---");
            // Limit content preview to first 1000 chars
            if content.len() > 1000 {
                println!("{}...", &content[..1000]);
                println!("\n[Content truncated, {} total bytes]", content.len());
            } else {
                println!("{}", content);
            }
        }
    } else {
        println!("Document not found.");
    }
}

/// Print sources list.
pub fn print_sources(result: &ListSourcesResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        if result.sources.is_empty() {
            println!("No sources indexed.");
            return;
        }

        println!(
            "{:<40} {:<10} {:<8} {:<8} PATH",
            "SOURCE ID", "TYPE", "DOCS", "WATCH"
        );
        println!("{}", "-".repeat(100));

        for source in &result.sources {
            let id_display = if source.source_id.len() > 38 {
                format!("{}...", &source.source_id[..35])
            } else {
                source.source_id.clone()
            };

            let path_display = if source.path.len() > 40 {
                format!("...{}", &source.path[source.path.len() - 37..])
            } else {
                source.path.clone()
            };

            println!(
                "{:<40} {:<10} {:<8} {:<8} {}",
                id_display,
                source.source_type,
                source.document_count,
                if source.watching { "yes" } else { "no" },
                path_display
            );
        }

        println!("\nTotal: {} sources", result.sources.len());
    }
}

/// Print remove source result.
pub fn print_remove_result(result: &RemoveSourceResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if result.success {
        println!("Removed {} documents", result.documents_removed);
        println!("{}", result.message);
    } else {
        println!("Failed to remove source");
        println!("{}", result.message);
    }
}

/// Print index statistics.
pub fn print_stats(result: &IndexStats, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        println!("Index Statistics");
        println!("{}", "=".repeat(40));
        println!("Sources:            {}", result.source_count);
        println!("Documents:          {}", result.document_count);
        println!("Chunks:             {}", result.chunk_count);
        println!("Storage:            {} bytes", result.storage_bytes);
        println!("Embedding Dim:      {}", result.embedding_dimension);
        println!("Storage Backend:    {}", result.storage_backend);
        println!("Embedding Provider: {}", result.embedding_provider);
        if result.uptime_secs > 0 {
            println!("Uptime:             {}s", result.uptime_secs);
        }
    }
}

/// Print clustering results.
pub fn print_cluster_results(result: &alloy::mcp::ClusterDocumentsResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        println!("Clustering Results");
        println!("{}", "=".repeat(60));
        println!(
            "Algorithm: {}  |  Total Documents: {}  |  Clusters: {}",
            result.algorithm, result.total_documents, result.metrics.num_clusters
        );
        println!(
            "Silhouette Score: {:.3}  |  Outliers: {}",
            result.metrics.silhouette_score, result.metrics.num_outliers
        );
        println!();

        if result.clusters.is_empty() {
            println!("No clusters found. Index more documents first.");
            return;
        }

        for cluster in &result.clusters {
            println!(
                "Cluster {} ({} docs): {}",
                cluster.cluster_id, cluster.size, cluster.label
            );
            if !cluster.keywords.is_empty() {
                println!("  Keywords: {}", cluster.keywords.join(", "));
            }
            if !cluster.representative_docs.is_empty() {
                println!(
                    "  Representatives: {}",
                    cluster.representative_docs[..cluster.representative_docs.len().min(3)]
                        .join(", ")
                );
            }
            println!("  Coherence: {:.3}", cluster.coherence_score);
            println!();
        }

        if !result.outliers.is_empty() {
            println!(
                "Outliers: {} documents not assigned to any cluster",
                result.outliers.len()
            );
            if result.outliers.len() <= 5 {
                for outlier in &result.outliers {
                    println!("  - {}", outlier);
                }
            } else {
                for outlier in result.outliers.iter().take(5) {
                    println!("  - {}", outlier);
                }
                println!("  ... and {} more", result.outliers.len() - 5);
            }
        }
    }
}
