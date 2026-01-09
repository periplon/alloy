//! Text diff computation module.
//!
//! Provides utilities for computing and displaying differences
//! between document versions.

use serde::{Deserialize, Serialize};

/// Statistics about a diff.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiffStats {
    /// Number of lines added.
    pub lines_added: usize,
    /// Number of lines removed.
    pub lines_removed: usize,
    /// Number of lines unchanged.
    pub lines_unchanged: usize,
    /// Number of characters added.
    pub chars_added: usize,
    /// Number of characters removed.
    pub chars_removed: usize,
}

/// A single change in the diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffChange {
    /// Line was unchanged.
    Equal { line: String },
    /// Line was added.
    Insert { line: String },
    /// Line was deleted.
    Delete { line: String },
}

/// Result of computing a diff between two texts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    /// Individual changes.
    pub changes: Vec<DiffChange>,
    /// Statistics about the diff.
    pub stats: DiffStats,
}

/// Unified diff format output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDiff {
    /// Header line for old version.
    pub old_header: String,
    /// Header line for new version.
    pub new_header: String,
    /// Hunks of changes.
    pub hunks: Vec<DiffHunk>,
    /// Statistics.
    pub stats: DiffStats,
}

/// A hunk in a unified diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    /// Starting line in old file.
    pub old_start: usize,
    /// Number of lines from old file.
    pub old_count: usize,
    /// Starting line in new file.
    pub new_start: usize,
    /// Number of lines from new file.
    pub new_count: usize,
    /// Lines in this hunk.
    pub lines: Vec<String>,
}

impl std::fmt::Display for UnifiedDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "--- {}", self.old_header)?;
        writeln!(f, "+++ {}", self.new_header)?;

        for hunk in &self.hunks {
            writeln!(
                f,
                "@@ -{},{} +{},{} @@",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            )?;
            for line in &hunk.lines {
                writeln!(f, "{}", line)?;
            }
        }

        Ok(())
    }
}

/// Compute line-by-line diff between two texts.
pub fn compute_diff(old: &str, new: &str) -> DiffResult {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Use Myers' diff algorithm (simplified LCS-based approach)
    let changes = compute_lcs_diff(&old_lines, &new_lines);

    let mut stats = DiffStats::default();
    for change in &changes {
        match change {
            DiffChange::Equal { line } => {
                stats.lines_unchanged += 1;
                // Don't count chars for unchanged lines
                let _ = line;
            }
            DiffChange::Insert { line } => {
                stats.lines_added += 1;
                stats.chars_added += line.len();
            }
            DiffChange::Delete { line } => {
                stats.lines_removed += 1;
                stats.chars_removed += line.len();
            }
        }
    }

    DiffResult { changes, stats }
}

/// Compute unified diff format.
pub fn compute_unified_diff(
    old: &str,
    new: &str,
    old_label: &str,
    new_label: &str,
    context_lines: usize,
) -> UnifiedDiff {
    let diff_result = compute_diff(old, new);
    let changes = &diff_result.changes;

    let mut hunks = Vec::new();
    let mut current_hunk: Option<DiffHunk> = None;
    let mut old_line = 1usize;
    let mut new_line = 1usize;

    for (i, change) in changes.iter().enumerate() {
        let is_change = !matches!(change, DiffChange::Equal { .. });

        // Check if we need to start a new hunk
        if is_change && current_hunk.is_none() {
            // Start new hunk with context
            let context_start = i.saturating_sub(context_lines);
            let mut hunk = DiffHunk {
                old_start: old_line.saturating_sub(i - context_start),
                old_count: 0,
                new_start: new_line.saturating_sub(i - context_start),
                new_count: 0,
                lines: Vec::new(),
            };

            // Add context lines before
            for change_item in changes.iter().take(i).skip(context_start) {
                if let DiffChange::Equal { line } = change_item {
                    hunk.lines.push(format!(" {}", line));
                    hunk.old_count += 1;
                    hunk.new_count += 1;
                }
            }

            current_hunk = Some(hunk);
        }

        // Add line to current hunk if active
        if let Some(ref mut hunk) = current_hunk {
            match change {
                DiffChange::Equal { line } => {
                    hunk.lines.push(format!(" {}", line));
                    hunk.old_count += 1;
                    hunk.new_count += 1;
                    old_line += 1;
                    new_line += 1;
                }
                DiffChange::Insert { line } => {
                    hunk.lines.push(format!("+{}", line));
                    hunk.new_count += 1;
                    new_line += 1;
                }
                DiffChange::Delete { line } => {
                    hunk.lines.push(format!("-{}", line));
                    hunk.old_count += 1;
                    old_line += 1;
                }
            }

            // Check if we should end the hunk
            let remaining_context = changes[i + 1..]
                .iter()
                .take(context_lines * 2 + 1)
                .filter(|c| !matches!(c, DiffChange::Equal { .. }))
                .count();

            if remaining_context == 0 && matches!(change, DiffChange::Equal { .. }) {
                // Count trailing context
                let trailing = changes[i + 1..]
                    .iter()
                    .take_while(|c| matches!(c, DiffChange::Equal { .. }))
                    .count();

                if trailing >= context_lines || i == changes.len() - 1 {
                    // End the hunk
                    if let Some(finished_hunk) = current_hunk.take() {
                        hunks.push(finished_hunk);
                    }
                }
            }
        } else {
            // Not in a hunk, just track line numbers
            match change {
                DiffChange::Equal { .. } => {
                    old_line += 1;
                    new_line += 1;
                }
                DiffChange::Insert { .. } => {
                    new_line += 1;
                }
                DiffChange::Delete { .. } => {
                    old_line += 1;
                }
            }
        }
    }

    // Don't forget the last hunk
    if let Some(hunk) = current_hunk {
        hunks.push(hunk);
    }

    UnifiedDiff {
        old_header: old_label.to_string(),
        new_header: new_label.to_string(),
        hunks,
        stats: diff_result.stats,
    }
}

/// Compute diff using LCS (Longest Common Subsequence) algorithm.
fn compute_lcs_diff(old: &[&str], new: &[&str]) -> Vec<DiffChange> {
    let m = old.len();
    let n = new.len();

    // Build LCS table
    let mut lcs = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if old[i - 1] == new[j - 1] {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
            } else {
                lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
            }
        }
    }

    // Backtrack to build diff
    let mut changes = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old[i - 1] == new[j - 1] {
            changes.push(DiffChange::Equal {
                line: old[i - 1].to_string(),
            });
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            changes.push(DiffChange::Insert {
                line: new[j - 1].to_string(),
            });
            j -= 1;
        } else {
            changes.push(DiffChange::Delete {
                line: old[i - 1].to_string(),
            });
            i -= 1;
        }
    }

    changes.reverse();
    changes
}

/// Compute delta operations for version storage.
pub fn compute_delta_operations(
    old: &str,
    new: &str,
) -> Vec<super::storage::DeltaOperation> {
    use super::storage::DeltaOperation;

    let diff = compute_diff(old, new);
    let mut operations = Vec::new();
    let mut position = 0usize;

    for change in diff.changes {
        match change {
            DiffChange::Equal { line } => {
                position += line.len() + 1; // +1 for newline
            }
            DiffChange::Insert { line } => {
                operations.push(DeltaOperation::Insert {
                    position,
                    text: format!("{}\n", line),
                });
                position += line.len() + 1;
            }
            DiffChange::Delete { line } => {
                operations.push(DeltaOperation::Delete {
                    position,
                    length: line.len() + 1,
                });
                // Don't advance position since we deleted
            }
        }
    }

    // Optimize: merge consecutive operations of the same type
    // This is left as an optimization for future work

    operations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_diff() {
        let old = "line1\nline2\nline3";
        let new = "line1\nmodified\nline3";

        let result = compute_diff(old, new);
        assert_eq!(result.stats.lines_unchanged, 2);
        assert_eq!(result.stats.lines_added, 1);
        assert_eq!(result.stats.lines_removed, 1);
    }

    #[test]
    fn test_unified_diff() {
        let old = "line1\nline2\nline3";
        let new = "line1\nnew line\nline3";

        let unified = compute_unified_diff(old, new, "old.txt", "new.txt", 3);
        let output = unified.to_string();

        assert!(output.contains("--- old.txt"));
        assert!(output.contains("+++ new.txt"));
        assert!(output.contains("-line2"));
        assert!(output.contains("+new line"));
    }

    #[test]
    fn test_no_changes() {
        let text = "same\ntext\nhere";
        let result = compute_diff(text, text);

        assert_eq!(result.stats.lines_unchanged, 3);
        assert_eq!(result.stats.lines_added, 0);
        assert_eq!(result.stats.lines_removed, 0);
    }

    #[test]
    fn test_all_new() {
        let old = "";
        let new = "line1\nline2";

        let result = compute_diff(old, new);
        assert_eq!(result.stats.lines_added, 2);
        assert_eq!(result.stats.lines_removed, 0);
    }

    #[test]
    fn test_all_deleted() {
        let old = "line1\nline2";
        let new = "";

        let result = compute_diff(old, new);
        assert_eq!(result.stats.lines_added, 0);
        assert_eq!(result.stats.lines_removed, 2);
    }

    #[test]
    fn test_delta_operations() {
        let old = "Hello, World!";
        let new = "Hello, Rust!";

        let ops = compute_delta_operations(old, new);
        assert!(!ops.is_empty());
    }
}
