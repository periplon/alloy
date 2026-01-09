//! Output formatting for CLI commands.
//!
//! This module handles formatting output as either JSON or human-readable text.

use alloy::mcp::{
    CreateBackupResponse, DocumentDetails, ExportDocumentsResponse, ImportDocumentsResponse,
    IndexPathResponse, IndexStats, ListBackupsResponse, ListSourcesResponse, RemoveSourceResponse,
    RestoreBackupResponse, SearchResponse,
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

/// Print backup result.
pub fn print_backup_result(result: &CreateBackupResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if result.success {
        println!("Backup Created Successfully");
        println!("{}", "=".repeat(40));
        println!("Backup ID:  {}", result.backup_id);
        println!("Path:       {}", result.path);
        println!("Documents:  {}", result.document_count);
        println!("Size:       {} bytes", result.size_bytes);
        println!("Duration:   {}ms", result.duration_ms);
    } else {
        println!("Backup Failed");
        println!("{}", result.message);
    }
}

/// Print restore result.
pub fn print_restore_result(result: &RestoreBackupResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if result.success {
        println!("Restore Completed Successfully");
        println!("{}", "=".repeat(40));
        println!("Backup ID:   {}", result.backup_id);
        println!("Documents:   {}", result.documents_restored);
        println!("Chunks:      {}", result.chunks_restored);
        println!("Duration:    {}ms", result.duration_ms);
    } else {
        println!("Restore Failed");
        println!("{}", result.message);
    }
}

/// Print export result.
pub fn print_export_result(result: &ExportDocumentsResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if result.success {
        println!("Export Completed Successfully");
        println!("{}", "=".repeat(40));
        println!("Output Path: {}", result.path);
        println!("Documents:   {}", result.document_count);
        println!("Size:        {} bytes", result.size_bytes);
        println!("Duration:    {}ms", result.duration_ms);
    } else {
        println!("Export Failed");
        println!("{}", result.message);
    }
}

/// Print import result.
pub fn print_import_result(result: &ImportDocumentsResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else if result.success {
        println!("Import Completed Successfully");
        println!("{}", "=".repeat(40));
        println!("Documents:   {}", result.documents_imported);
        println!("Chunks:      {}", result.chunks_imported);
        println!("Duration:    {}ms", result.duration_ms);
    } else {
        println!("Import Failed");
        println!("{}", result.message);
    }
}

/// Print list backups result.
pub fn print_list_backups_result(result: &ListBackupsResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
    } else {
        if result.backups.is_empty() {
            println!("No backups found.");
            return;
        }

        println!("Available Backups");
        println!("{}", "=".repeat(80));
        println!(
            "{:<38} {:<20} {:>8} {:>10}",
            "BACKUP ID", "CREATED", "DOCS", "SIZE"
        );
        println!("{}", "-".repeat(80));

        for backup in &result.backups {
            let id_display = if backup.backup_id.len() > 36 {
                format!("{}...", &backup.backup_id[..33])
            } else {
                backup.backup_id.clone()
            };

            let created = backup.created_at.format("%Y-%m-%d %H:%M:%S");

            let size_display = if backup.size_bytes > 1024 * 1024 {
                format!("{:.1} MB", backup.size_bytes as f64 / (1024.0 * 1024.0))
            } else if backup.size_bytes > 1024 {
                format!("{:.1} KB", backup.size_bytes as f64 / 1024.0)
            } else {
                format!("{} B", backup.size_bytes)
            };

            println!(
                "{:<38} {:<20} {:>8} {:>10}",
                id_display, created, backup.document_count, size_display
            );

            if let Some(desc) = &backup.description {
                println!("  Description: {}", desc);
            }
        }

        println!();
        println!("Total: {} backups", result.backups.len());
    }
}

// ============================================================================
// GTD, Calendar, Knowledge, Query, Ontology Output Formatters
// ============================================================================

use super::types::{CalendarResult, GtdResult, KnowledgeResult, OntologyResult, QueryResult};

/// Truncate a string to a maximum length with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

/// Format a date-time value from JSON.
fn format_datetime_value(value: &serde_json::Value) -> String {
    if let Some(s) = value.as_str() {
        // Try to extract just the date and time portion
        if s.len() >= 16 {
            s[..16].replace('T', " ")
        } else {
            s.to_string()
        }
    } else {
        "-".to_string()
    }
}

/// Print GTD command result.
pub fn print_gtd_result(result: &GtdResult, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
        return;
    }

    if !result.success {
        println!("Error: {}", result.message);
        return;
    }

    println!("{}\n", result.message);

    // Get the data value for formatting
    let data = result.data_value();

    // Try to format data based on its structure
    if let Some(array) = data.as_array() {
        if array.is_empty() {
            println!("No items found.");
            return;
        }

        // Check if it's a list of projects
        if array.first().and_then(|v| v.get("outcome")).is_some() {
            println!("{:<36} {:<10} {:<20}", "ID", "STATUS", "NAME");
            println!("{}", "-".repeat(70));
            for item in array {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("-");
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                println!(
                    "{:<36} {:<10} {}",
                    truncate(id, 34),
                    status,
                    truncate(name, 40)
                );
            }
        }
        // Check if it's a list of tasks
        else if array.first().and_then(|v| v.get("description")).is_some() {
            println!(
                "{:<36} {:<8} {:<8} {:<30}",
                "ID", "STATUS", "PRIORITY", "DESCRIPTION"
            );
            println!("{}", "-".repeat(90));
            for item in array {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("-");
                let priority = item.get("priority").and_then(|v| v.as_str()).unwrap_or("-");
                let desc = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!(
                    "{:<36} {:<8} {:<8} {}",
                    truncate(id, 34),
                    status,
                    priority,
                    truncate(desc, 40)
                );
            }
        }
        // Check if it's a list of waiting items
        else if array.first().and_then(|v| v.get("delegated_to")).is_some() {
            println!(
                "{:<36} {:<15} {:<12} {:<30}",
                "ID", "DELEGATED TO", "STATUS", "DESCRIPTION"
            );
            println!("{}", "-".repeat(100));
            for item in array {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let delegated = item
                    .get("delegated_to")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("-");
                let desc = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!(
                    "{:<36} {:<15} {:<12} {}",
                    truncate(id, 34),
                    truncate(delegated, 13),
                    status,
                    truncate(desc, 35)
                );
            }
        }
        // Check if it's a list of entities (ontology)
        else if array.first().and_then(|v| v.get("entity_type")).is_some() {
            println!("{:<36} {:<15} {:<30}", "ID", "TYPE", "NAME");
            println!("{}", "-".repeat(85));
            for item in array {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let etype = item
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                println!(
                    "{:<36} {:<15} {}",
                    truncate(id, 34),
                    etype,
                    truncate(name, 40)
                );
            }
        }
        // Check if it's a list of commitments
        else if array
            .first()
            .and_then(|v| v.get("commitment_type"))
            .is_some()
        {
            println!(
                "{:<36} {:<10} {:<10} {:<30}",
                "ID", "TYPE", "STATUS", "DESCRIPTION"
            );
            println!("{}", "-".repeat(90));
            for item in array {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let ctype = item
                    .get("commitment_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let status = item.get("status").and_then(|v| v.as_str()).unwrap_or("-");
                let desc = item
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!(
                    "{:<36} {:<10} {:<10} {}",
                    truncate(id, 34),
                    ctype,
                    status,
                    truncate(desc, 35)
                );
            }
        }
        // Generic array output
        else {
            for item in array {
                println!("{}", serde_json::to_string_pretty(item).unwrap());
            }
        }
    }
    // Single object output
    else if data.is_object() {
        // Check if it's a weekly review report
        if data.get("period").is_some() {
            print_weekly_review_data(&data);
        }
        // Check if it's horizon overview
        else if data.get("horizons").is_some() {
            print_horizon_data(&data);
        }
        // Check if it's attention metrics
        else if data.get("total_time_minutes").is_some() {
            print_attention_data(&data);
        }
        // Check if it's a dependency graph
        else if data.get("nodes").is_some() && data.get("edges").is_some() {
            print_dependency_data(&data);
        }
        // Check if it's a single project
        else if data.get("outcome").is_some() {
            print_single_project(&data);
        }
        // Check if it's a single task
        else if data.get("description").is_some() && data.get("contexts").is_some() {
            print_single_task(&data);
        }
        // Generic object
        else {
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
    }
}

fn print_weekly_review_data(data: &serde_json::Value) {
    println!("=== Weekly Review ===\n");

    if let Some(period) = data.get("period") {
        let start = period.get("start").and_then(|v| v.as_str()).unwrap_or("-");
        let end = period.get("end").and_then(|v| v.as_str()).unwrap_or("-");
        println!("Period: {} to {}", start, end);
    }

    println!("\nSUMMARY");
    if let Some(completed) = data.get("completed_count").and_then(|v| v.as_u64()) {
        println!("  Tasks Completed:  {}", completed);
    }
    if let Some(active) = data.get("active_projects").and_then(|v| v.as_u64()) {
        println!("  Active Projects:  {}", active);
    }
    if let Some(inbox) = data.get("inbox_count").and_then(|v| v.as_u64()) {
        println!("  Inbox Items:      {}", inbox);
    }

    if let Some(stalled) = data.get("stalled_projects").and_then(|v| v.as_array()) {
        if !stalled.is_empty() {
            println!("\nSTALLED PROJECTS (need attention)");
            for p in stalled {
                let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                let days = p.get("days_stalled").and_then(|v| v.as_u64()).unwrap_or(0);
                println!("  - {} ({} days)", name, days);
            }
        }
    }

    if let Some(waiting) = data.get("overdue_waiting").and_then(|v| v.as_array()) {
        if !waiting.is_empty() {
            println!("\nOVERDUE WAITING ITEMS");
            for w in waiting {
                let desc = w.get("description").and_then(|v| v.as_str()).unwrap_or("-");
                let person = w
                    .get("delegated_to")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!("  - {} (from {})", desc, person);
            }
        }
    }

    if let Some(suggestions) = data.get("suggestions").and_then(|v| v.as_array()) {
        if !suggestions.is_empty() {
            println!("\nRECOMMENDATIONS");
            for s in suggestions {
                if let Some(text) = s.as_str() {
                    println!("  • {}", text);
                }
            }
        }
    }
}

fn print_horizon_data(data: &serde_json::Value) {
    println!("=== GTD Horizons ===\n");

    if let Some(horizons) = data.get("horizons").and_then(|v| v.as_object()) {
        for (level, items) in horizons {
            let level_name = match level.as_str() {
                "Runway" => "Runway (Current Actions)",
                "H10k" => "10,000 ft (Projects)",
                "H20k" => "20,000 ft (Areas of Focus)",
                "H30k" => "30,000 ft (Goals)",
                "H40k" => "40,000 ft (Vision)",
                "H50k" => "50,000 ft (Purpose)",
                _ => level,
            };
            println!("{}", level_name);
            println!("{}", "-".repeat(40));

            if let Some(arr) = items.as_array() {
                if arr.is_empty() {
                    println!("  (no items)");
                } else {
                    for item in arr {
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                        println!("  • {}", name);
                    }
                }
            }
            println!();
        }
    }
}

fn print_attention_data(data: &serde_json::Value) {
    println!("=== Attention Analysis ===\n");

    if let Some(total) = data.get("total_time_minutes").and_then(|v| v.as_f64()) {
        let hours = total / 60.0;
        println!("Total Time:  {:.1} hours ({:.0} minutes)", hours, total);
    }

    if let Some(by_area) = data.get("by_area").and_then(|v| v.as_object()) {
        if !by_area.is_empty() {
            println!("\nBy Area:");
            for (area, time) in by_area {
                let minutes = time.as_f64().unwrap_or(0.0);
                let hours = minutes / 60.0;
                println!("  {:<20} {:.1}h ({:.0}m)", area, hours, minutes);
            }
        }
    }

    if let Some(by_project) = data.get("by_project").and_then(|v| v.as_object()) {
        if !by_project.is_empty() {
            println!("\nBy Project:");
            for (project, time) in by_project {
                let minutes = time.as_f64().unwrap_or(0.0);
                let hours = minutes / 60.0;
                println!(
                    "  {:<30} {:.1}h ({:.0}m)",
                    truncate(project, 28),
                    hours,
                    minutes
                );
            }
        }
    }
}

fn print_dependency_data(data: &serde_json::Value) {
    println!("=== Dependency Graph ===\n");

    if let Some(nodes) = data.get("nodes").and_then(|v| v.as_array()) {
        println!("Nodes: {}", nodes.len());
    }
    if let Some(edges) = data.get("edges").and_then(|v| v.as_array()) {
        println!("Dependencies: {}", edges.len());
    }

    if let Some(critical) = data.get("critical_path").and_then(|v| v.as_array()) {
        if !critical.is_empty() {
            println!("\nCritical Path:");
            for (i, item) in critical.iter().enumerate() {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                println!("  {}. {}", i + 1, name);
            }
        }
    }

    if let Some(blockers) = data.get("blockers").and_then(|v| v.as_array()) {
        if !blockers.is_empty() {
            println!("\nBlocking Items:");
            for blocker in blockers {
                let name = blocker.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                let blocked = blocker
                    .get("blocks_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                println!("  - {} (blocks {} items)", name, blocked);
            }
        }
    }
}

fn print_single_project(data: &serde_json::Value) {
    println!("=== Project Details ===\n");

    let name = data.get("name").and_then(|v| v.as_str()).unwrap_or("-");
    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("-");
    let status = data.get("status").and_then(|v| v.as_str()).unwrap_or("-");
    let outcome = data.get("outcome").and_then(|v| v.as_str()).unwrap_or("-");

    println!("Name:     {}", name);
    println!("ID:       {}", id);
    println!("Status:   {}", status);
    println!("Outcome:  {}", outcome);

    if let Some(area) = data.get("area").and_then(|v| v.as_str()) {
        println!("Area:     {}", area);
    }
    if let Some(goal) = data.get("goal").and_then(|v| v.as_str()) {
        println!("Goal:     {}", goal);
    }
    if let Some(next) = data.get("next_action") {
        if let Some(desc) = next.get("description").and_then(|v| v.as_str()) {
            println!("Next:     {}", desc);
        }
    }
}

fn print_single_task(data: &serde_json::Value) {
    println!("=== Task Details ===\n");

    let desc = data
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("-");
    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("-");
    let status = data.get("status").and_then(|v| v.as_str()).unwrap_or("-");

    println!("Description: {}", desc);
    println!("ID:          {}", id);
    println!("Status:      {}", status);

    if let Some(priority) = data.get("priority").and_then(|v| v.as_str()) {
        println!("Priority:    {}", priority);
    }
    if let Some(contexts) = data.get("contexts").and_then(|v| v.as_array()) {
        let ctx_str: Vec<&str> = contexts.iter().filter_map(|v| v.as_str()).collect();
        if !ctx_str.is_empty() {
            println!("Contexts:    {}", ctx_str.join(", "));
        }
    }
    if let Some(energy) = data.get("energy_level").and_then(|v| v.as_str()) {
        println!("Energy:      {}", energy);
    }
    if let Some(duration) = data.get("estimated_duration").and_then(|v| v.as_u64()) {
        println!("Duration:    {} minutes", duration);
    }
    if let Some(due) = data.get("due_date") {
        println!("Due:         {}", format_datetime_value(due));
    }
    if let Some(project) = data.get("project_id").and_then(|v| v.as_str()) {
        println!("Project:     {}", project);
    }
}

/// Print Calendar command result.
pub fn print_calendar_result(result: &CalendarResult, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
        return;
    }

    if !result.success {
        println!("Error: {}", result.message);
        return;
    }

    println!("{}\n", result.message);

    // Get the data value for formatting
    let data = result.data_value();

    // Try to format data based on its structure
    if let Some(array) = data.as_array() {
        if array.is_empty() {
            println!("No events found.");
            return;
        }

        // Check if it's a list of events
        if array.first().and_then(|v| v.get("title")).is_some() {
            println!("{:<18} {:<18} {:<30}", "START", "END", "TITLE");
            println!("{}", "-".repeat(70));
            for event in array {
                let start = event
                    .get("start")
                    .map(format_datetime_value)
                    .unwrap_or_else(|| "-".to_string());
                let end = event
                    .get("end")
                    .map(format_datetime_value)
                    .unwrap_or_else(|| "-".to_string());
                let title = event.get("title").and_then(|v| v.as_str()).unwrap_or("-");
                println!("{:<18} {:<18} {}", start, end, truncate(title, 40));
                if let Some(loc) = event.get("location").and_then(|v| v.as_str()) {
                    if !loc.is_empty() {
                        println!("                                      Location: {}", loc);
                    }
                }
            }
        }
        // Check if it's free time slots
        else if array.first().and_then(|v| v.get("start_time")).is_some() {
            println!("{:<18} {:<18} {:<10}", "START", "END", "DURATION");
            println!("{}", "-".repeat(50));
            for slot in array {
                let start = slot
                    .get("start_time")
                    .map(format_datetime_value)
                    .unwrap_or_else(|| "-".to_string());
                let end = slot
                    .get("end_time")
                    .map(format_datetime_value)
                    .unwrap_or_else(|| "-".to_string());
                let duration = slot
                    .get("duration_minutes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                println!("{:<18} {:<18} {} min", start, end, duration);
            }
        }
        // Check if it's conflicts
        else if array.first().and_then(|v| v.get("event1")).is_some() {
            println!("Schedule Conflicts:\n");
            for (i, conflict) in array.iter().enumerate() {
                let e1 = conflict
                    .get("event1")
                    .and_then(|v| v.get("title"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let e2 = conflict
                    .get("event2")
                    .and_then(|v| v.get("title"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!("{}. \"{}\" conflicts with \"{}\"", i + 1, e1, e2);
            }
        } else {
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
    }
    // Single event
    else if data.is_object() && data.get("title").is_some() {
        print_single_event(&data);
    } else if !data.is_null() {
        println!("{}", serde_json::to_string_pretty(&data).unwrap());
    }
}

fn print_single_event(data: &serde_json::Value) {
    println!("=== Event Details ===\n");

    let title = data.get("title").and_then(|v| v.as_str()).unwrap_or("-");
    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("-");
    let start = data
        .get("start")
        .map(format_datetime_value)
        .unwrap_or_else(|| "-".to_string());
    let end = data
        .get("end")
        .map(format_datetime_value)
        .unwrap_or_else(|| "-".to_string());

    println!("Title:    {}", title);
    println!("ID:       {}", id);
    println!("Start:    {}", start);
    println!("End:      {}", end);

    if let Some(etype) = data.get("event_type").and_then(|v| v.as_str()) {
        println!("Type:     {}", etype);
    }
    if let Some(loc) = data.get("location").and_then(|v| v.as_str()) {
        if !loc.is_empty() {
            println!("Location: {}", loc);
        }
    }
    if let Some(participants) = data.get("participants").and_then(|v| v.as_array()) {
        let parts: Vec<&str> = participants.iter().filter_map(|v| v.as_str()).collect();
        if !parts.is_empty() {
            println!("Participants: {}", parts.join(", "));
        }
    }
    if let Some(notes) = data.get("notes").and_then(|v| v.as_str()) {
        if !notes.is_empty() {
            println!("Notes:    {}", notes);
        }
    }
}

/// Print Knowledge command result.
pub fn print_knowledge_result(result: &KnowledgeResult, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
        return;
    }

    if !result.success {
        println!("Error: {}", result.message);
        return;
    }

    println!("{}\n", result.message);

    // Get the data value for formatting
    let data = result.data_value();

    // Try to format data based on its structure
    if let Some(array) = data.as_array() {
        if array.is_empty() {
            println!("No results found.");
            return;
        }

        // Check if it's a list of entities
        if array.first().and_then(|v| v.get("entity_type")).is_some() {
            println!("{:<36} {:<15} {:<30}", "ID", "TYPE", "NAME");
            println!("{}", "-".repeat(85));
            for entity in array {
                let id = entity.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let etype = entity
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let name = entity.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                println!(
                    "{:<36} {:<15} {}",
                    truncate(id, 34),
                    etype,
                    truncate(name, 40)
                );
            }
        }
        // Check if it's connected entities with path info
        else if array.first().and_then(|v| v.get("entity")).is_some() {
            println!("Connected Entities:\n");
            for item in array {
                if let Some(entity) = item.get("entity") {
                    let name = entity.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                    let etype = entity
                        .get("entity_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("-");
                    let path_len = item
                        .get("path_length")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    println!("  {} [{}] (distance: {})", name, etype, path_len);
                }
            }
        } else {
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
    }
    // Single entity or topic
    else if data.is_object() {
        if data.get("entity").is_some() {
            // Entity with relationships
            if let Some(entity) = data.get("entity") {
                print_single_entity(entity);
            }
            if let Some(rels) = data.get("relationships").and_then(|v| v.as_array()) {
                if !rels.is_empty() {
                    println!("\nRelationships ({}):", rels.len());
                    for rel in rels {
                        let rel_type = rel
                            .get("relationship_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("-");
                        let target = rel
                            .get("target_entity_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("-");
                        println!("  {} -> {}", rel_type, truncate(target, 30));
                    }
                }
            }
        } else if data.get("entity_type").is_some() {
            print_single_entity(&data);
        } else if data.get("topic").is_some() {
            if let Some(topic) = data.get("topic") {
                print_single_entity(topic);
            }
            if let Some(count) = data.get("related_entities").and_then(|v| v.as_u64()) {
                println!("\nRelated Entities: {}", count);
            }
        } else {
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
    }
}

fn print_single_entity(data: &serde_json::Value) {
    println!("=== Entity Details ===\n");

    let name = data.get("name").and_then(|v| v.as_str()).unwrap_or("-");
    let id = data.get("id").and_then(|v| v.as_str()).unwrap_or("-");
    let etype = data
        .get("entity_type")
        .and_then(|v| v.as_str())
        .unwrap_or("-");

    println!("Name:     {}", name);
    println!("ID:       {}", id);
    println!("Type:     {}", etype);

    if let Some(aliases) = data.get("aliases").and_then(|v| v.as_array()) {
        let alias_str: Vec<&str> = aliases.iter().filter_map(|v| v.as_str()).collect();
        if !alias_str.is_empty() {
            println!("Aliases:  {}", alias_str.join(", "));
        }
    }
    if let Some(meta) = data.get("metadata").and_then(|v| v.as_object()) {
        if !meta.is_empty() {
            println!("Metadata:");
            for (key, value) in meta {
                if let Some(s) = value.as_str() {
                    println!("  {}: {}", key, s);
                }
            }
        }
    }
}

/// Print Query command result.
pub fn print_query_result(result: &QueryResult, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
        return;
    }

    println!("Query: {}", result.query);
    println!("Mode:  {}\n", result.mode);

    if !result.success {
        println!("Error: {}", result.message);
        return;
    }

    println!("{}\n", result.message);

    // Get the data value for formatting
    let data = result.data_value();

    // Format results based on mode and data
    if let Some(array) = data.as_array() {
        if array.is_empty() {
            println!("No results found.");
            return;
        }

        // Generic entity list
        println!("{:<36} {:<15} {:<30}", "ID", "TYPE", "NAME");
        println!("{}", "-".repeat(85));
        for entity in array {
            let id = entity.get("id").and_then(|v| v.as_str()).unwrap_or("-");
            let etype = entity
                .get("entity_type")
                .and_then(|v| v.as_str())
                .unwrap_or("-");
            let name = entity.get("name").and_then(|v| v.as_str()).unwrap_or("-");
            println!(
                "{:<36} {:<15} {}",
                truncate(id, 34),
                etype,
                truncate(name, 40)
            );
        }
    } else if !data.is_null() {
        println!("{}", serde_json::to_string_pretty(&data).unwrap());
    }
}

/// Print Ontology command result.
pub fn print_ontology_result(result: &OntologyResult, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(result).unwrap());
        return;
    }

    if !result.success {
        println!("Error: {}", result.message);
        return;
    }

    println!("{}\n", result.message);

    // Get the data value for formatting
    let data = result.data_value();

    // Check for stats
    if data.get("entity_count").is_some() {
        let entities = data.get("entity_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let rels = data
            .get("relationship_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        println!("Ontology Statistics");
        println!("{}", "=".repeat(40));
        println!("Entities:       {}", entities);
        println!("Relationships:  {}", rels);

        if let Some(by_type) = data.get("entities_by_type").and_then(|v| v.as_object()) {
            println!("\nEntities by Type:");
            for (etype, count) in by_type {
                let c = count.as_u64().unwrap_or(0);
                println!("  {:<15} {}", etype, c);
            }
        }
        return;
    }

    // Try to format data based on its structure
    if let Some(array) = data.as_array() {
        if array.is_empty() {
            println!("No items found.");
            return;
        }

        // Check if it's entities
        if array.first().and_then(|v| v.get("entity_type")).is_some() {
            println!("{:<36} {:<15} {:<30}", "ID", "TYPE", "NAME");
            println!("{}", "-".repeat(85));
            for entity in array {
                let id = entity.get("id").and_then(|v| v.as_str()).unwrap_or("-");
                let etype = entity
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let name = entity.get("name").and_then(|v| v.as_str()).unwrap_or("-");
                println!(
                    "{:<36} {:<15} {}",
                    truncate(id, 34),
                    etype,
                    truncate(name, 40)
                );
            }
        }
        // Check if it's relationships
        else if array
            .first()
            .and_then(|v| v.get("relationship_type"))
            .is_some()
        {
            println!("{:<36} {:<20} {:<36}", "SOURCE", "RELATIONSHIP", "TARGET");
            println!("{}", "-".repeat(95));
            for rel in array {
                let source = rel
                    .get("source_entity_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let rel_type = rel
                    .get("relationship_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                let target = rel
                    .get("target_entity_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-");
                println!(
                    "{:<36} {:<20} {}",
                    truncate(source, 34),
                    rel_type,
                    truncate(target, 34)
                );
            }
        } else {
            println!("{}", serde_json::to_string_pretty(&data).unwrap());
        }
    }
    // Single entity
    else if data.is_object() && data.get("entity_type").is_some() {
        print_single_entity(&data);
    }
    // Single relationship
    else if data.is_object() && data.get("relationship_type").is_some() {
        let source = data
            .get("source_entity_id")
            .and_then(|v| v.as_str())
            .unwrap_or("-");
        let rel_type = data
            .get("relationship_type")
            .and_then(|v| v.as_str())
            .unwrap_or("-");
        let target = data
            .get("target_entity_id")
            .and_then(|v| v.as_str())
            .unwrap_or("-");
        println!("Relationship: {} -> {} -> {}", source, rel_type, target);
    } else if !data.is_null() {
        println!("{}", serde_json::to_string_pretty(&data).unwrap());
    }
}
