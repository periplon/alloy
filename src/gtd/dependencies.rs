//! Dependency Graph for GTD.
//!
//! This module provides dependency visualization and analysis for tasks
//! and projects, including critical path detection and blocker identification.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::gtd::types::{Task, TaskFilter, TaskStatus};
use crate::gtd::TaskManager;
use crate::ontology::{EmbeddedOntologyStore, OntologyStore, RelationType, RelationshipFilter};
use schemars::JsonSchema;

// ============================================================================
// Dependency Types
// ============================================================================

/// Parameters for dependency graph generation.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DependencyParams {
    /// Focus on a specific project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Include completed items.
    #[serde(default)]
    pub include_completed: bool,
    /// Maximum depth to traverse.
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Include critical path analysis.
    #[serde(default = "default_true")]
    pub include_critical_path: bool,
    /// Include blocker analysis.
    #[serde(default = "default_true")]
    pub include_blockers: bool,
    /// Output format for visualization.
    #[serde(default)]
    pub output_format: OutputFormat,
}

fn default_max_depth() -> usize {
    10
}

fn default_true() -> bool {
    true
}

/// Output format for dependency visualization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    /// JSON structure.
    #[default]
    Json,
    /// Mermaid diagram syntax.
    Mermaid,
    /// DOT/Graphviz format.
    Dot,
    /// Simple text representation.
    Text,
}

impl Default for DependencyParams {
    fn default() -> Self {
        Self {
            project_id: None,
            include_completed: false,
            max_depth: 10,
            include_critical_path: true,
            include_blockers: true,
            output_format: OutputFormat::Json,
        }
    }
}

/// Complete dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// All nodes in the graph.
    pub nodes: Vec<DependencyNode>,
    /// All edges (dependencies) in the graph.
    pub edges: Vec<DependencyEdge>,
    /// Critical path (longest dependency chain).
    pub critical_path: CriticalPath,
    /// Current blockers.
    pub blockers: Vec<BlockerInfo>,
    /// Graph statistics.
    pub stats: GraphStats,
    /// Visualization in requested format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visualization: Option<String>,
    /// When this graph was generated.
    pub generated_at: DateTime<Utc>,
}

/// A node in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNode {
    /// Unique identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Node type (task, project, milestone).
    pub node_type: NodeType,
    /// Current status.
    pub status: String,
    /// Is this node on the critical path?
    pub on_critical_path: bool,
    /// Is this node currently blocked?
    pub is_blocked: bool,
    /// Is this node blocking others?
    pub is_blocking: bool,
    /// Number of dependencies (incoming edges).
    pub dependency_count: usize,
    /// Number of dependents (outgoing edges).
    pub dependent_count: usize,
    /// Estimated duration in minutes (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_minutes: Option<u32>,
    /// Due date (if set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_date: Option<DateTime<Utc>>,
    /// Project this belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

/// Type of node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    /// A task.
    Task,
    /// A project.
    Project,
    /// A milestone.
    Milestone,
    /// An external dependency.
    External,
}

/// An edge (dependency) in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Source node ID (the dependency).
    pub from: String,
    /// Target node ID (the dependent).
    pub to: String,
    /// Type of dependency.
    pub dependency_type: DependencyType,
    /// Is this edge on the critical path?
    pub on_critical_path: bool,
    /// Description of the dependency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Type of dependency relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    /// Finish-to-start: predecessor must finish before successor starts.
    FinishToStart,
    /// Start-to-start: predecessor must start before successor starts.
    StartToStart,
    /// Finish-to-finish: predecessor must finish before successor finishes.
    FinishToFinish,
    /// Start-to-finish: predecessor must start before successor finishes.
    StartToFinish,
    /// Blocks: explicit blocker relationship.
    Blocks,
    /// Waiting for: waiting on something/someone.
    WaitingFor,
}

/// Critical path analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPath {
    /// Nodes on the critical path in order.
    pub path: Vec<String>,
    /// Node names for display.
    pub path_names: Vec<String>,
    /// Total length of the path (number of nodes).
    pub length: usize,
    /// Total estimated duration in minutes.
    pub total_estimated_minutes: Option<u32>,
    /// Earliest possible completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub earliest_completion: Option<DateTime<Utc>>,
    /// Description.
    pub description: String,
}

/// Information about a blocker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockerInfo {
    /// The blocking task/item.
    pub blocker: BlockerItem,
    /// Items being blocked.
    pub blocked_items: Vec<BlockedItem>,
    /// Days this has been blocking.
    pub days_blocking: u32,
    /// Severity level.
    pub severity: BlockerSeverity,
    /// Impact description.
    pub impact: String,
    /// Suggested resolution.
    pub suggestion: String,
}

/// A blocking item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockerItem {
    /// Item ID.
    pub id: String,
    /// Item name.
    pub name: String,
    /// Item type.
    pub item_type: String,
    /// Current status.
    pub status: String,
    /// Assigned to (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigned_to: Option<String>,
}

/// An item that is blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedItem {
    /// Item ID.
    pub id: String,
    /// Item name.
    pub name: String,
    /// Item type.
    pub item_type: String,
    /// How it's blocked.
    pub blocked_reason: String,
}

/// Severity of a blocker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockerSeverity {
    /// Low impact.
    Low,
    /// Medium impact.
    Medium,
    /// High impact.
    High,
    /// Critical - blocks multiple items or critical path.
    Critical,
}

/// Graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes.
    pub total_nodes: usize,
    /// Total number of edges.
    pub total_edges: usize,
    /// Number of root nodes (no dependencies).
    pub root_nodes: usize,
    /// Number of leaf nodes (no dependents).
    pub leaf_nodes: usize,
    /// Number of blocked items.
    pub blocked_count: usize,
    /// Number of blocking items.
    pub blocker_count: usize,
    /// Average dependencies per node.
    pub avg_dependencies: f32,
    /// Maximum dependency chain length.
    pub max_chain_length: usize,
    /// Has cycles (circular dependencies).
    pub has_cycles: bool,
}

// ============================================================================
// Dependency Manager
// ============================================================================

/// Manages dependency graph generation and analysis.
pub struct DependencyManager {
    store: Arc<RwLock<EmbeddedOntologyStore>>,
    task_manager: TaskManager,
}

impl DependencyManager {
    /// Create a new dependency manager.
    pub fn new(store: Arc<RwLock<EmbeddedOntologyStore>>) -> Self {
        Self {
            task_manager: TaskManager::new(store.clone()),
            store,
        }
    }

    /// Generate a dependency graph.
    pub async fn generate(&self, params: DependencyParams) -> Result<DependencyGraph> {
        // Get all tasks
        let mut task_filter = TaskFilter::default();
        if !params.include_completed {
            task_filter.status = Some(TaskStatus::Next);
        }
        if let Some(ref project_id) = params.project_id {
            task_filter.project_id = Some(project_id.clone());
        }

        let tasks = self.task_manager.list(task_filter).await?;

        // Get relationships from ontology store
        let store = self.store.read().await;
        let all_relationships = store
            .list_relationships(RelationshipFilter::default())
            .await?;
        drop(store);

        // Build nodes
        let mut nodes: Vec<DependencyNode> = Vec::new();
        let mut node_ids: HashSet<String> = HashSet::new();

        for task in &tasks {
            nodes.push(DependencyNode {
                id: task.id.clone(),
                name: task.description.clone(),
                node_type: NodeType::Task,
                status: format!("{:?}", task.status),
                on_critical_path: false,
                is_blocked: false,
                is_blocking: false,
                dependency_count: 0,
                dependent_count: 0,
                estimated_minutes: task.estimated_minutes,
                due_date: task.due_date,
                project_id: task.project_id.clone(),
            });
            node_ids.insert(task.id.clone());
        }

        // Build edges from relationships
        let mut edges: Vec<DependencyEdge> = Vec::new();
        let mut blocked_by: HashMap<String, Vec<String>> = HashMap::new();
        let mut blocking: HashMap<String, Vec<String>> = HashMap::new();

        for rel in &all_relationships {
            // Only include edges where both nodes are in our graph
            if !node_ids.contains(&rel.source_entity_id)
                || !node_ids.contains(&rel.target_entity_id)
            {
                continue;
            }

            let dep_type = match rel.relationship_type {
                RelationType::BlockedBy => DependencyType::Blocks,
                RelationType::DependsOn => DependencyType::FinishToStart,
                RelationType::WaitingOn => DependencyType::WaitingFor,
                _ => continue,
            };

            edges.push(DependencyEdge {
                from: rel.source_entity_id.clone(),
                to: rel.target_entity_id.clone(),
                dependency_type: dep_type,
                on_critical_path: false,
                description: None,
            });

            // Track blocking relationships
            blocked_by
                .entry(rel.target_entity_id.clone())
                .or_default()
                .push(rel.source_entity_id.clone());
            blocking
                .entry(rel.source_entity_id.clone())
                .or_default()
                .push(rel.target_entity_id.clone());
        }

        // Update node dependency counts
        for node in &mut nodes {
            node.dependency_count = blocked_by.get(&node.id).map(|v| v.len()).unwrap_or(0);
            node.dependent_count = blocking.get(&node.id).map(|v| v.len()).unwrap_or(0);
            node.is_blocked = node.dependency_count > 0;
            node.is_blocking = node.dependent_count > 0;
        }

        // Calculate critical path
        let critical_path = if params.include_critical_path {
            self.calculate_critical_path(&nodes, &edges)
        } else {
            CriticalPath {
                path: Vec::new(),
                path_names: Vec::new(),
                length: 0,
                total_estimated_minutes: None,
                earliest_completion: None,
                description: "Not calculated".to_string(),
            }
        };

        // Mark critical path nodes and edges
        let critical_set: HashSet<String> = critical_path.path.iter().cloned().collect();
        for node in &mut nodes {
            node.on_critical_path = critical_set.contains(&node.id);
        }
        for edge in &mut edges {
            edge.on_critical_path =
                critical_set.contains(&edge.from) && critical_set.contains(&edge.to);
        }

        // Analyze blockers
        let blockers = if params.include_blockers {
            self.analyze_blockers(&nodes, &blocking, &blocked_by, &tasks)
        } else {
            Vec::new()
        };

        // Calculate statistics
        let stats = self.calculate_stats(&nodes, &edges, &blocking, &blocked_by);

        // Generate visualization
        let visualization = match params.output_format {
            OutputFormat::Json => None,
            OutputFormat::Mermaid => Some(self.to_mermaid(&nodes, &edges)),
            OutputFormat::Dot => Some(self.to_dot(&nodes, &edges)),
            OutputFormat::Text => Some(self.to_text(&nodes, &edges, &critical_path)),
        };

        Ok(DependencyGraph {
            nodes,
            edges,
            critical_path,
            blockers,
            stats,
            visualization,
            generated_at: Utc::now(),
        })
    }

    /// Get blockers for a specific task or project.
    pub async fn get_blockers(&self, item_id: &str) -> Result<Vec<BlockerInfo>> {
        let graph = self
            .generate(DependencyParams {
                include_blockers: true,
                ..Default::default()
            })
            .await?;

        Ok(graph
            .blockers
            .into_iter()
            .filter(|b| {
                b.blocker.id == item_id || b.blocked_items.iter().any(|bi| bi.id == item_id)
            })
            .collect())
    }

    /// Get the critical path for a project.
    pub async fn get_critical_path(&self, project_id: &str) -> Result<CriticalPath> {
        let graph = self
            .generate(DependencyParams {
                project_id: Some(project_id.to_string()),
                include_critical_path: true,
                ..Default::default()
            })
            .await?;

        Ok(graph.critical_path)
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    fn calculate_critical_path(
        &self,
        nodes: &[DependencyNode],
        edges: &[DependencyEdge],
    ) -> CriticalPath {
        if nodes.is_empty() {
            return CriticalPath {
                path: Vec::new(),
                path_names: Vec::new(),
                length: 0,
                total_estimated_minutes: None,
                earliest_completion: None,
                description: "No nodes in graph".to_string(),
            };
        }

        // Build adjacency list
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for edge in edges {
            adj.entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }

        // Find all paths using DFS and track the longest
        let node_map: HashMap<String, &DependencyNode> =
            nodes.iter().map(|n| (n.id.clone(), n)).collect();

        let mut longest_path: Vec<String> = Vec::new();
        let mut longest_duration: u32 = 0;

        // Start from nodes with no incoming edges (roots)
        let targets: HashSet<String> = edges.iter().map(|e| e.to.clone()).collect();
        let roots: Vec<&String> = nodes
            .iter()
            .map(|n| &n.id)
            .filter(|id| !targets.contains(*id))
            .collect();

        for root in roots {
            let mut visited: HashSet<String> = HashSet::new();
            let mut current_path: Vec<String> = Vec::new();

            Self::dfs_longest_path(
                root,
                &adj,
                &node_map,
                &mut visited,
                &mut current_path,
                &mut longest_path,
                &mut longest_duration,
            );
        }

        // Get names for the path
        let path_names: Vec<String> = longest_path
            .iter()
            .filter_map(|id| node_map.get(id).map(|n| n.name.clone()))
            .collect();

        // Calculate total estimated duration
        let total_estimated: Option<u32> = if longest_path
            .iter()
            .all(|id| node_map.get(id).and_then(|n| n.estimated_minutes).is_some())
        {
            Some(
                longest_path
                    .iter()
                    .filter_map(|id| node_map.get(id).and_then(|n| n.estimated_minutes))
                    .sum(),
            )
        } else {
            None
        };

        let description = if longest_path.is_empty() {
            "No dependency chains found".to_string()
        } else {
            format!(
                "Critical path has {} items{}",
                longest_path.len(),
                total_estimated
                    .map(|m| format!(", estimated {} minutes total", m))
                    .unwrap_or_default()
            )
        };

        let length = path_names.len();

        CriticalPath {
            path: longest_path,
            path_names,
            length,
            total_estimated_minutes: total_estimated,
            earliest_completion: None, // Would need more complex scheduling
            description,
        }
    }

    fn dfs_longest_path(
        node: &str,
        adj: &HashMap<String, Vec<String>>,
        node_map: &HashMap<String, &DependencyNode>,
        visited: &mut HashSet<String>,
        current_path: &mut Vec<String>,
        longest_path: &mut Vec<String>,
        longest_duration: &mut u32,
    ) {
        if visited.contains(node) {
            return; // Cycle detected
        }

        visited.insert(node.to_string());
        current_path.push(node.to_string());

        // Calculate current path duration
        let current_duration: u32 = current_path
            .iter()
            .filter_map(|id| node_map.get(id).and_then(|n| n.estimated_minutes))
            .sum();

        // Check if this path is longer
        if current_path.len() > longest_path.len()
            || (current_path.len() == longest_path.len() && current_duration > *longest_duration)
        {
            *longest_path = current_path.clone();
            *longest_duration = current_duration;
        }

        // Continue DFS
        if let Some(neighbors) = adj.get(node) {
            for neighbor in neighbors {
                Self::dfs_longest_path(
                    neighbor,
                    adj,
                    node_map,
                    visited,
                    current_path,
                    longest_path,
                    longest_duration,
                );
            }
        }

        current_path.pop();
        visited.remove(node);
    }

    fn analyze_blockers(
        &self,
        nodes: &[DependencyNode],
        blocking: &HashMap<String, Vec<String>>,
        _blocked_by: &HashMap<String, Vec<String>>,
        tasks: &[Task],
    ) -> Vec<BlockerInfo> {
        let mut blockers = Vec::new();
        let node_map: HashMap<String, &DependencyNode> =
            nodes.iter().map(|n| (n.id.clone(), n)).collect();
        let task_map: HashMap<String, &Task> = tasks.iter().map(|t| (t.id.clone(), t)).collect();

        // Find all blocking nodes
        for (blocker_id, blocked_ids) in blocking {
            if blocked_ids.is_empty() {
                continue;
            }

            let blocker_node = match node_map.get(blocker_id) {
                Some(n) => n,
                None => continue,
            };

            let blocker_task = task_map.get(blocker_id);

            let blocked_items: Vec<BlockedItem> = blocked_ids
                .iter()
                .filter_map(|id| {
                    node_map.get(id).map(|n| BlockedItem {
                        id: n.id.clone(),
                        name: n.name.clone(),
                        item_type: format!("{:?}", n.node_type),
                        blocked_reason: "Waiting for dependency to complete".to_string(),
                    })
                })
                .collect();

            let days_blocking = blocker_task
                .map(|t| (Utc::now() - t.created_at).num_days() as u32)
                .unwrap_or(0);

            let severity = if blocked_items.len() > 3 || blocker_node.on_critical_path {
                BlockerSeverity::Critical
            } else if blocked_items.len() > 1 {
                BlockerSeverity::High
            } else if days_blocking > 7 {
                BlockerSeverity::Medium
            } else {
                BlockerSeverity::Low
            };

            let impact = format!(
                "Blocking {} item(s){}",
                blocked_items.len(),
                if blocker_node.on_critical_path {
                    " on critical path"
                } else {
                    ""
                }
            );

            let suggestion = if blocker_node.status.contains("Done") {
                "Blocker is completed - update dependencies".to_string()
            } else {
                format!("Prioritize completing: {}", blocker_node.name)
            };

            blockers.push(BlockerInfo {
                blocker: BlockerItem {
                    id: blocker_node.id.clone(),
                    name: blocker_node.name.clone(),
                    item_type: format!("{:?}", blocker_node.node_type),
                    status: blocker_node.status.clone(),
                    assigned_to: None,
                },
                blocked_items,
                days_blocking,
                severity,
                impact,
                suggestion,
            });
        }

        // Sort by severity
        blockers.sort_by(|a, b| {
            let ord = |s: &BlockerSeverity| match s {
                BlockerSeverity::Critical => 0,
                BlockerSeverity::High => 1,
                BlockerSeverity::Medium => 2,
                BlockerSeverity::Low => 3,
            };
            ord(&a.severity).cmp(&ord(&b.severity))
        });

        blockers
    }

    fn calculate_stats(
        &self,
        nodes: &[DependencyNode],
        edges: &[DependencyEdge],
        _blocking: &HashMap<String, Vec<String>>,
        _blocked_by: &HashMap<String, Vec<String>>,
    ) -> GraphStats {
        let total_nodes = nodes.len();
        let total_edges = edges.len();

        let root_nodes = nodes.iter().filter(|n| n.dependency_count == 0).count();
        let leaf_nodes = nodes.iter().filter(|n| n.dependent_count == 0).count();

        let blocked_count = nodes.iter().filter(|n| n.is_blocked).count();
        let blocker_count = nodes.iter().filter(|n| n.is_blocking).count();

        let avg_dependencies = if total_nodes > 0 {
            total_edges as f32 / total_nodes as f32
        } else {
            0.0
        };

        // Detect cycles using DFS
        let has_cycles = self.detect_cycles(nodes, edges);

        // Calculate max chain length (same as critical path length)
        let max_chain_length = nodes.iter().filter(|n| n.on_critical_path).count().max(1);

        GraphStats {
            total_nodes,
            total_edges,
            root_nodes,
            leaf_nodes,
            blocked_count,
            blocker_count,
            avg_dependencies,
            max_chain_length,
            has_cycles,
        }
    }

    fn detect_cycles(&self, nodes: &[DependencyNode], edges: &[DependencyEdge]) -> bool {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for edge in edges {
            adj.entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut rec_stack: HashSet<String> = HashSet::new();

        for node in nodes {
            if Self::has_cycle_from(&node.id, &adj, &mut visited, &mut rec_stack) {
                return true;
            }
        }

        false
    }

    fn has_cycle_from(
        node: &str,
        adj: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        if rec_stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }

        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = adj.get(node) {
            for neighbor in neighbors {
                if Self::has_cycle_from(neighbor, adj, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    fn to_mermaid(&self, nodes: &[DependencyNode], edges: &[DependencyEdge]) -> String {
        let mut output = String::from("graph TD\n");

        for node in nodes {
            let shape_start = if node.on_critical_path { "((" } else { "[" };
            let shape_end = if node.on_critical_path { "))" } else { "]" };
            let status_suffix = if node.is_blocked {
                " 🔴"
            } else if node.is_blocking {
                " 🟡"
            } else {
                ""
            };

            output.push_str(&format!(
                "    {}{}\"{}{}\"{};\n",
                node.id.replace('-', "_"),
                shape_start,
                node.name,
                status_suffix,
                shape_end
            ));
        }

        output.push('\n');

        for edge in edges {
            let arrow = if edge.on_critical_path { "==>" } else { "-->" };
            output.push_str(&format!(
                "    {} {} {};\n",
                edge.from.replace('-', "_"),
                arrow,
                edge.to.replace('-', "_")
            ));
        }

        output
    }

    fn to_dot(&self, nodes: &[DependencyNode], edges: &[DependencyEdge]) -> String {
        let mut output = String::from("digraph dependencies {\n");
        output.push_str("    rankdir=TB;\n");
        output.push_str("    node [shape=box];\n\n");

        for node in nodes {
            let color = if node.on_critical_path {
                "red"
            } else if node.is_blocked {
                "orange"
            } else {
                "black"
            };
            output.push_str(&format!(
                "    \"{}\" [label=\"{}\" color=\"{}\"];\n",
                node.id, node.name, color
            ));
        }

        output.push('\n');

        for edge in edges {
            let style = if edge.on_critical_path {
                "bold"
            } else {
                "solid"
            };
            output.push_str(&format!(
                "    \"{}\" -> \"{}\" [style=\"{}\"];\n",
                edge.from, edge.to, style
            ));
        }

        output.push_str("}\n");
        output
    }

    fn to_text(
        &self,
        nodes: &[DependencyNode],
        edges: &[DependencyEdge],
        critical_path: &CriticalPath,
    ) -> String {
        let mut output = String::new();

        output.push_str("=== Dependency Graph ===\n\n");

        output.push_str("NODES:\n");
        for node in nodes {
            let flags = format!(
                "{}{}{}",
                if node.on_critical_path { "[CP] " } else { "" },
                if node.is_blocked { "[BLOCKED] " } else { "" },
                if node.is_blocking { "[BLOCKING] " } else { "" }
            );
            output.push_str(&format!("  - {} {}{}\n", node.id, flags, node.name));
        }

        output.push_str("\nDEPENDENCIES:\n");
        for edge in edges {
            output.push_str(&format!("  {} --> {}\n", edge.from, edge.to));
        }

        output.push_str("\nCRITICAL PATH:\n");
        if critical_path.path.is_empty() {
            output.push_str("  (none)\n");
        } else {
            output.push_str(&format!("  {}\n", critical_path.path_names.join(" -> ")));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_manager() -> DependencyManager {
        let store = Arc::new(RwLock::new(EmbeddedOntologyStore::new()));
        DependencyManager::new(store)
    }

    #[tokio::test]
    async fn test_generate_empty_graph() {
        let manager = create_test_manager().await;
        let graph = manager.generate(DependencyParams::default()).await.unwrap();

        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.edges.len(), 0);
    }

    #[tokio::test]
    async fn test_output_formats() {
        let manager = create_test_manager().await;

        // Test Mermaid format
        let graph = manager
            .generate(DependencyParams {
                output_format: OutputFormat::Mermaid,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(graph.visualization.is_some());

        // Test DOT format
        let graph = manager
            .generate(DependencyParams {
                output_format: OutputFormat::Dot,
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(graph.visualization.is_some());
    }

    #[test]
    fn test_blocker_severity() {
        let severity = BlockerSeverity::Critical;
        assert_eq!(severity, BlockerSeverity::Critical);
    }

    #[test]
    fn test_dependency_type() {
        let dep = DependencyType::FinishToStart;
        assert_eq!(dep, DependencyType::FinishToStart);
    }
}
