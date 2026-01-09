//! Action item detection for tasks, commitments, and follow-ups.
//!
//! This module identifies actionable items from natural language text,
//! supporting:
//! - Task markers: "TODO:", "ACTION:", "TASK:", "[ ]"
//! - Commitment detection: "I will", "I'll", "I promise"
//! - Follow-up detection: "Follow up on", "Check with"
//! - Delegation detection: "Ask X to", "Have Y"
//! - Reminder patterns: "Don't forget to", "Remember to"

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// An action item detected in text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    /// Unique identifier for the action.
    pub id: String,
    /// The action description.
    pub description: String,
    /// Type of action.
    pub action_type: ActionType,
    /// Priority level.
    pub priority: Priority,
    /// Estimated energy level required.
    pub energy_level: EnergyLevel,
    /// Confidence of detection.
    pub confidence: f32,
}

/// A detected action from text analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAction {
    /// The action description extracted from text.
    pub description: String,
    /// The original text that contained the action.
    pub source_text: String,
    /// Type of action detected.
    pub action_type: ActionType,
    /// Priority level (if detected).
    pub priority: Priority,
    /// Energy level estimate.
    pub energy_level: EnergyLevel,
    /// For commitments: who made the commitment.
    pub commitment_by: Option<String>,
    /// For commitments: who the commitment is to.
    pub commitment_to: Option<String>,
    /// For follow-ups: who to follow up with.
    pub follow_up_with: Option<String>,
    /// Confidence of detection (0.0-1.0).
    pub confidence: f32,
    /// Character offset where the action starts.
    pub start_offset: usize,
    /// Character offset where the action ends.
    pub end_offset: usize,
}

/// Type of action item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    /// A task to be completed.
    Task,
    /// A commitment or promise.
    Commitment,
    /// A follow-up item (checking in, waiting for response).
    FollowUp,
    /// A reminder or note-to-self.
    Reminder,
}

/// Type of commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitmentType {
    /// A commitment made by the speaker.
    Made,
    /// A commitment received from someone else.
    Received,
}

/// Priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Low priority.
    Low,
    /// Normal/medium priority.
    #[default]
    Normal,
    /// High priority.
    High,
    /// Critical/urgent priority.
    Critical,
}

/// Energy level required for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnergyLevel {
    /// Low energy, routine task.
    Low,
    /// Medium energy, focused work.
    #[default]
    Medium,
    /// High energy, demanding task.
    High,
}

// ============================================================================
// Action Detector
// ============================================================================

/// Detects action items from text.
pub struct ActionDetector {
    /// Patterns for task markers.
    task_patterns: Vec<TaskPattern>,
    /// Patterns for commitments.
    commitment_patterns: Vec<CommitmentPattern>,
    /// Patterns for follow-ups.
    follow_up_patterns: Vec<FollowUpPattern>,
    /// Patterns for reminders.
    reminder_patterns: Vec<ReminderPattern>,
    /// Priority indicators.
    priority_indicators: PriorityIndicators,
}

struct TaskPattern {
    regex: regex::Regex,
    confidence: f32,
}

struct CommitmentPattern {
    regex: regex::Regex,
    commitment_type: CommitmentType,
    confidence: f32,
}

struct FollowUpPattern {
    regex: regex::Regex,
    confidence: f32,
}

struct ReminderPattern {
    regex: regex::Regex,
    confidence: f32,
}

struct PriorityIndicators {
    critical: Vec<String>,
    high: Vec<String>,
    low: Vec<String>,
}

impl Default for ActionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionDetector {
    /// Create a new action detector with default patterns.
    pub fn new() -> Self {
        Self {
            task_patterns: Self::default_task_patterns(),
            commitment_patterns: Self::default_commitment_patterns(),
            follow_up_patterns: Self::default_follow_up_patterns(),
            reminder_patterns: Self::default_reminder_patterns(),
            priority_indicators: Self::default_priority_indicators(),
        }
    }

    fn default_task_patterns() -> Vec<TaskPattern> {
        let patterns = [
            // Explicit markers - match anywhere in text (allow prefix like URGENT)
            (r"(?i)\b(?:TODO|TO-DO|TO DO)\s*[:;]?\s*([^\n]+)", 0.95),
            (r"(?i)\b(?:ACTION|ACTION ITEM)\s*[:;]?\s*([^\n]+)", 0.95),
            (r"(?i)\b(?:TASK)\s*[:;]?\s*([^\n]+)", 0.95),
            (r"(?i)\b(?:FIXME|FIX)\s*[:;]?\s*([^\n]+)", 0.9),
            (r"(?i)\b(?:HACK|XXX)\s*[:;]?\s*([^\n]+)", 0.85),

            // Checkbox patterns - match at start of line or after newline
            (r"(?m)^\s*\[\s*\]\s*([^\n]+)", 0.9),  // [ ] task
            (r"(?m)^\s*-\s*\[\s*\]\s*([^\n]+)", 0.9),  // - [ ] task (Markdown)

            // Bullet/numbered lists with action verbs
            (r"(?i)(?:^|\n)\s*[-*•]\s*((?:need to|must|should|have to|gotta|gonna)\s+.+?)(?:\n|$)", 0.75),
            (r"(?i)(?:^|\n)\s*\d+[.)]\s*((?:need to|must|should|have to)\s+.+?)(?:\n|$)", 0.75),

            // Action verb at start of sentence
            (r"(?i)(?:^|\n|[.!?]\s+)((?:implement|create|build|develop|design|write|fix|update|review|test|check|verify|complete|finish|prepare|send|submit|schedule|arrange|organize|plan|research|investigate|analyze|evaluate|assess|determine|decide|consider|resolve|address|handle|process|configure|setup|set up|install|deploy|migrate|refactor|optimize|improve|enhance|add|remove|delete|modify|change|edit|correct|adjust|ensure|make sure|validate|confirm|document|draft|outline|define|specify|establish|clarify)\s+.{10,80})(?:[.!?]|\n|$)", 0.7),
        ];

        patterns
            .into_iter()
            .filter_map(|(p, c)| {
                regex::Regex::new(p).ok().map(|r| TaskPattern {
                    regex: r,
                    confidence: c,
                })
            })
            .collect()
    }

    fn default_commitment_patterns() -> Vec<CommitmentPattern> {
        let patterns = [
            // Commitments made (first person)
            (r"(?i)\b(I(?:'ll| will| am going to| shall| promise to| commit to| guarantee to| pledge to)\s+.{5,80})(?:[.!?]|\n|$)", CommitmentType::Made, 0.85),
            (r"(?i)\b(I(?:'m| am)\s+(?:going to|planning to|committed to)\s+.{5,80})(?:[.!?]|\n|$)", CommitmentType::Made, 0.85),
            (r"(?i)\b(I can\s+(?:do|handle|take care of|manage|complete|finish)\s+.{5,80})(?:[.!?]|\n|$)", CommitmentType::Made, 0.8),
            (r"(?i)\b(Let me\s+.{5,60})(?:[.!?]|\n|$)", CommitmentType::Made, 0.75),
            (r"(?i)\b(I(?:'ll| will) (?:get back to|follow up|send|email|call|contact|reach out to)\s+.{3,50})(?:[.!?]|\n|$)", CommitmentType::Made, 0.85),

            // Commitments received (second/third person)
            (r"(?i)\b((?:You|He|She|They)(?:'ll| will| promised to| agreed to| committed to)\s+.{5,80})(?:[.!?]|\n|$)", CommitmentType::Received, 0.85),
            (r"(?i)\b((?:You|He|She|They)\s+(?:said|mentioned|indicated)\s+(?:you|he|she|they)(?:'d| would)\s+.{5,60})(?:[.!?]|\n|$)", CommitmentType::Received, 0.8),

            // Request patterns (often imply commitments)
            (r"(?i)\b(Can you\s+(?:please\s+)?(?:send|provide|share|update|review|check|confirm)\s+.{3,60})\??", CommitmentType::Received, 0.7),
            (r"(?i)\b(Please\s+(?:send|provide|share|update|review|check|confirm|complete|finish|submit)\s+.{3,60})(?:[.!?]|\n|$)", CommitmentType::Received, 0.75),
        ];

        patterns
            .into_iter()
            .filter_map(|(p, ct, c)| {
                regex::Regex::new(p).ok().map(|r| CommitmentPattern {
                    regex: r,
                    commitment_type: ct,
                    confidence: c,
                })
            })
            .collect()
    }

    fn default_follow_up_patterns() -> Vec<FollowUpPattern> {
        let patterns = [
            // Explicit follow-up
            (r"(?i)\b(follow[- ]up\s+(?:with|on)\s+.{3,60})(?:[.!?]|\n|$)", 0.9),
            (r"(?i)\b(check\s+(?:in\s+)?with\s+.{3,40})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(reach out to\s+.{3,40})(?:[.!?]|\n|$)", 0.8),
            (r"(?i)\b(ping\s+.{3,30})(?:[.!?]|\n|$)", 0.75),
            (r"(?i)\b(circle back\s+(?:with|on|to)\s+.{3,50})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(touch base with\s+.{3,40})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(sync(?:\s+up)? with\s+.{3,40})(?:[.!?]|\n|$)", 0.8),

            // Waiting patterns
            (r"(?i)\b(waiting (?:for|on)\s+.{3,60})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(need\s+(?:response|reply|answer|feedback|input)\s+from\s+.{3,40})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(pending\s+(?:response|reply|review|approval)\s+from\s+.{3,40})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(awaiting\s+.{3,50})(?:[.!?]|\n|$)", 0.85),

            // Ask/request patterns
            (r"(?i)\b(ask\s+(?:\w+\s+)?to\s+.{5,50})(?:[.!?]|\n|$)", 0.8),
            (r"(?i)\b(have\s+(?:\w+\s+)?(?:to\s+)?(?:review|check|look at|approve|sign off on)\s+.{3,50})(?:[.!?]|\n|$)", 0.75),
        ];

        patterns
            .into_iter()
            .filter_map(|(p, c)| {
                regex::Regex::new(p).ok().map(|r| FollowUpPattern {
                    regex: r,
                    confidence: c,
                })
            })
            .collect()
    }

    fn default_reminder_patterns() -> Vec<ReminderPattern> {
        let patterns = [
            // Explicit reminders
            (r"(?i)\b((?:don't forget|remember|remind me)\s+to\s+.{5,60})(?:[.!?]|\n|$)", 0.9),
            (r"(?i)\b(make sure\s+(?:to\s+)?.{5,60})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(note(?:\s+to\s+self)?:\s*.{5,80})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)(?:^|\n)\s*(?:NOTE|NB)\s*[:;]\s*(.{5,80})(?:\n|$)", 0.85),
            (r"(?i)\b(important:\s*.{5,80})(?:[.!?]|\n|$)", 0.8),

            // Don't/avoid patterns (negative reminders)
            (r"(?i)\b(don't\s+(?:forget|miss|skip|overlook)\s+.{5,50})(?:[.!?]|\n|$)", 0.85),
            (r"(?i)\b(be sure to\s+.{5,50})(?:[.!?]|\n|$)", 0.8),
        ];

        patterns
            .into_iter()
            .filter_map(|(p, c)| {
                regex::Regex::new(p).ok().map(|r| ReminderPattern {
                    regex: r,
                    confidence: c,
                })
            })
            .collect()
    }

    fn default_priority_indicators() -> PriorityIndicators {
        PriorityIndicators {
            critical: vec![
                "urgent".to_string(),
                "asap".to_string(),
                "immediately".to_string(),
                "critical".to_string(),
                "blocker".to_string(),
                "blocking".to_string(),
                "p0".to_string(),
                "priority 0".to_string(),
                "🚨".to_string(),
                "‼️".to_string(),
                "🔴".to_string(),
            ],
            high: vec![
                "important".to_string(),
                "high priority".to_string(),
                "p1".to_string(),
                "priority 1".to_string(),
                "time-sensitive".to_string(),
                "deadline".to_string(),
                "eod".to_string(),
                "end of day".to_string(),
                "today".to_string(),
                "🟠".to_string(),
                "⚠️".to_string(),
            ],
            low: vec![
                "low priority".to_string(),
                "p3".to_string(),
                "p4".to_string(),
                "priority 3".to_string(),
                "priority 4".to_string(),
                "when you have time".to_string(),
                "nice to have".to_string(),
                "optional".to_string(),
                "someday".to_string(),
                "eventually".to_string(),
                "later".to_string(),
                "🟢".to_string(),
            ],
        }
    }

    /// Detect all action items in the text.
    pub fn detect(&self, text: &str) -> Vec<DetectedAction> {
        let mut actions = Vec::new();

        // Detect tasks
        for pattern in &self.task_patterns {
            for cap in pattern.regex.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let description = Self::clean_description(m.as_str());
                    if description.len() >= 5 {
                        // Minimum length check
                        let full_match = cap.get(0).unwrap();
                        // Include both the preceding context and the full match for priority detection
                        let context = format!("{} {}", &text[..full_match.start()], full_match.as_str());
                        let priority = self.detect_priority(&description, &context);
                        let energy = Self::estimate_energy(&description);

                        actions.push(DetectedAction {
                            description: description.clone(),
                            source_text: full_match.as_str().to_string(),
                            action_type: ActionType::Task,
                            priority,
                            energy_level: energy,
                            commitment_by: None,
                            commitment_to: None,
                            follow_up_with: None,
                            confidence: pattern.confidence,
                            start_offset: full_match.start(),
                            end_offset: full_match.end(),
                        });
                    }
                }
            }
        }

        // Detect commitments
        for pattern in &self.commitment_patterns {
            for cap in pattern.regex.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let description = Self::clean_description(m.as_str());
                    if description.len() >= 5 {
                        let full_match = cap.get(0).unwrap();
                        let priority =
                            self.detect_priority(&description, &text[..full_match.start()]);

                        let (commitment_by, commitment_to) = match pattern.commitment_type {
                            CommitmentType::Made => (Some("self".to_string()), None),
                            CommitmentType::Received => {
                                let person = Self::extract_person_name(&description);
                                (person, Some("self".to_string()))
                            }
                        };

                        actions.push(DetectedAction {
                            description: description.clone(),
                            source_text: full_match.as_str().to_string(),
                            action_type: ActionType::Commitment,
                            priority,
                            energy_level: EnergyLevel::Medium,
                            commitment_by,
                            commitment_to,
                            follow_up_with: None,
                            confidence: pattern.confidence,
                            start_offset: full_match.start(),
                            end_offset: full_match.end(),
                        });
                    }
                }
            }
        }

        // Detect follow-ups
        for pattern in &self.follow_up_patterns {
            for cap in pattern.regex.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let description = Self::clean_description(m.as_str());
                    if description.len() >= 5 {
                        let full_match = cap.get(0).unwrap();
                        let priority =
                            self.detect_priority(&description, &text[..full_match.start()]);
                        let follow_up_with = Self::extract_person_name(&description);

                        actions.push(DetectedAction {
                            description: description.clone(),
                            source_text: full_match.as_str().to_string(),
                            action_type: ActionType::FollowUp,
                            priority,
                            energy_level: EnergyLevel::Low,
                            commitment_by: None,
                            commitment_to: None,
                            follow_up_with,
                            confidence: pattern.confidence,
                            start_offset: full_match.start(),
                            end_offset: full_match.end(),
                        });
                    }
                }
            }
        }

        // Detect reminders
        for pattern in &self.reminder_patterns {
            for cap in pattern.regex.captures_iter(text) {
                if let Some(m) = cap.get(1) {
                    let description = Self::clean_description(m.as_str());
                    if description.len() >= 5 {
                        let full_match = cap.get(0).unwrap();
                        let priority =
                            self.detect_priority(&description, &text[..full_match.start()]);

                        actions.push(DetectedAction {
                            description: description.clone(),
                            source_text: full_match.as_str().to_string(),
                            action_type: ActionType::Reminder,
                            priority,
                            energy_level: EnergyLevel::Low,
                            commitment_by: None,
                            commitment_to: None,
                            follow_up_with: None,
                            confidence: pattern.confidence,
                            start_offset: full_match.start(),
                            end_offset: full_match.end(),
                        });
                    }
                }
            }
        }

        // Sort by offset and remove duplicates/overlaps
        actions.sort_by_key(|a| a.start_offset);
        Self::remove_overlapping_actions(&mut actions);

        actions
    }

    /// Clean up a description string.
    fn clean_description(s: &str) -> String {
        s.trim()
            .trim_end_matches(&['.', '!', '?', ':', ';', ','][..])
            .trim()
            .to_string()
    }

    /// Detect priority from context.
    fn detect_priority(&self, description: &str, context: &str) -> Priority {
        let combined = format!("{} {}", context, description).to_lowercase();

        for indicator in &self.priority_indicators.critical {
            if combined.contains(&indicator.to_lowercase()) {
                return Priority::Critical;
            }
        }

        for indicator in &self.priority_indicators.high {
            if combined.contains(&indicator.to_lowercase()) {
                return Priority::High;
            }
        }

        for indicator in &self.priority_indicators.low {
            if combined.contains(&indicator.to_lowercase()) {
                return Priority::Low;
            }
        }

        Priority::Normal
    }

    /// Estimate energy level based on task description.
    fn estimate_energy(description: &str) -> EnergyLevel {
        let desc_lower = description.to_lowercase();

        // High energy tasks (creative, complex)
        let high_energy_words = [
            "design",
            "architect",
            "implement",
            "develop",
            "create",
            "build",
            "analyze",
            "research",
            "investigate",
            "debug",
            "troubleshoot",
            "optimize",
            "refactor",
            "write",
            "draft",
            "plan",
            "strategize",
            "present",
            "lead",
            "facilitate",
        ];

        // Low energy tasks (routine, simple)
        let low_energy_words = [
            "email",
            "send",
            "reply",
            "forward",
            "file",
            "organize",
            "sort",
            "update",
            "check",
            "confirm",
            "schedule",
            "book",
            "remind",
            "note",
            "log",
            "record",
            "approve",
            "sign",
            "submit",
            "pay",
        ];

        for word in &high_energy_words {
            if desc_lower.contains(word) {
                return EnergyLevel::High;
            }
        }

        for word in &low_energy_words {
            if desc_lower.contains(word) {
                return EnergyLevel::Low;
            }
        }

        EnergyLevel::Medium
    }

    /// Try to extract a person's name from text.
    fn extract_person_name(text: &str) -> Option<String> {
        // Simple pattern: capitalized words that look like names
        let name_pattern =
            regex::Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b").expect("Invalid regex");

        // Look for common patterns like "with John", "from Jane" - stop at non-name words
        let with_pattern =
            regex::Regex::new(r"(?i)(?:with|from|to|for|ask|contact|email|call|ping)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
                .expect("Invalid regex");

        if let Some(cap) = with_pattern.captures(text) {
            return Some(cap[1].to_string());
        }

        // Fall back to first capitalized name found
        if let Some(cap) = name_pattern.captures(text) {
            let name = &cap[1];
            // Filter out common non-name words
            let non_names = [
                "I",
                "The",
                "This",
                "That",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ];
            if !non_names.contains(&name) {
                return Some(name.to_string());
            }
        }

        None
    }

    /// Remove overlapping actions, keeping the highest confidence one.
    fn remove_overlapping_actions(actions: &mut Vec<DetectedAction>) {
        if actions.len() < 2 {
            return;
        }

        let mut i = 0;
        while i < actions.len() - 1 {
            let current_end = actions[i].end_offset;
            let next_start = actions[i + 1].start_offset;

            if next_start < current_end {
                // Overlap detected - keep the higher confidence one
                if actions[i].confidence >= actions[i + 1].confidence {
                    actions.remove(i + 1);
                } else {
                    actions.remove(i);
                }
            } else {
                i += 1;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_todo() {
        let detector = ActionDetector::new();
        let text = "TODO: Complete the quarterly report\nTODO: Review PR #123";
        let actions = detector.detect(text);

        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.action_type == ActionType::Task && a.description.contains("quarterly report")));
        assert!(actions[0].confidence > 0.9);
    }

    #[test]
    fn test_detect_checkbox() {
        let detector = ActionDetector::new();
        let text = "Tasks:\n[ ] Write documentation\n[ ] Update tests";
        let actions = detector.detect(text);

        // Should detect at least one checkbox task
        let checkbox_tasks: Vec<_> = actions.iter()
            .filter(|a| a.action_type == ActionType::Task)
            .collect();
        assert!(!checkbox_tasks.is_empty(), "Should detect checkbox tasks");
    }

    #[test]
    fn test_detect_action_item() {
        let detector = ActionDetector::new();
        let text = "ACTION: Send the proposal to the client\nACTION ITEM: Schedule team meeting";
        let actions = detector.detect(text);

        // Should detect at least one action item
        assert!(!actions.is_empty(), "Should detect action items");
    }

    #[test]
    fn test_detect_commitment_made() {
        let detector = ActionDetector::new();
        let text = "I'll send you the report by end of day. I will review the code tomorrow.";
        let actions = detector.detect(text);

        let commitments: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::Commitment)
            .collect();
        assert!(!commitments.is_empty());
    }

    #[test]
    fn test_detect_follow_up() {
        let detector = ActionDetector::new();
        let text = "Follow up with John about the project. Check in with Sarah next week.";
        let actions = detector.detect(text);

        let follow_ups: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::FollowUp)
            .collect();
        assert!(!follow_ups.is_empty());
        assert!(follow_ups[0].follow_up_with.is_some());
    }

    #[test]
    fn test_detect_reminder() {
        let detector = ActionDetector::new();
        let text = "Don't forget to submit the timesheet. Remember to call mom.";
        let actions = detector.detect(text);

        let reminders: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::Reminder)
            .collect();
        assert!(!reminders.is_empty());
    }

    #[test]
    fn test_detect_priority() {
        let detector = ActionDetector::new();
        let text = "URGENT TODO: Fix the production bug";
        let actions = detector.detect(text);

        // Find the urgent one
        let urgent: Vec<_> = actions
            .iter()
            .filter(|a| a.priority == Priority::Critical || a.priority == Priority::High)
            .collect();
        assert!(!urgent.is_empty(), "Should detect urgent/high priority task");
    }

    #[test]
    fn test_estimate_energy() {
        assert_eq!(ActionDetector::estimate_energy("design the new API"), EnergyLevel::High);
        assert_eq!(ActionDetector::estimate_energy("send an email to John"), EnergyLevel::Low);
        assert_eq!(ActionDetector::estimate_energy("do something"), EnergyLevel::Medium);
    }

    #[test]
    fn test_extract_person_name() {
        // Test explicit pattern with "with"
        let result = ActionDetector::extract_person_name("follow up with John Smith");
        assert!(result.is_some(), "Should extract name from 'with' pattern");
        assert!(result.unwrap().starts_with("John"), "Should extract John's name");

        // Test pattern with "email"
        let result = ActionDetector::extract_person_name("email Sarah");
        assert!(result.is_some(), "Should extract name from 'email' pattern");
    }

    #[test]
    fn test_no_false_positives() {
        let detector = ActionDetector::new();
        let text = "This is just a normal sentence. Nothing to do here. The weather is nice.";
        let actions = detector.detect(text);

        // Should have few or no detections
        assert!(actions.len() <= 1);
    }

    #[test]
    fn test_complex_document() {
        let detector = ActionDetector::new();
        let text = r#"
Meeting Notes - January 10, 2024

Attendees: John, Sarah, Mike

Discussion:
- Reviewed Q4 performance metrics
- Discussed roadmap for Q1

Action Items:
TODO: John to prepare budget proposal by Friday
ACTION: Sarah will schedule follow-up meeting
[ ] Update project timeline

Follow-ups:
- Follow up with marketing team about campaign results
- Check in with dev team on deployment status

Reminders:
- Don't forget to submit expense reports
- Remember to review the new policy document

Notes:
I'll send the summary to everyone by EOD.
        "#;

        let actions = detector.detect(text);

        // Should detect multiple types
        let tasks: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::Task)
            .collect();
        let follow_ups: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::FollowUp)
            .collect();
        let reminders: Vec<_> = actions
            .iter()
            .filter(|a| a.action_type == ActionType::Reminder)
            .collect();

        assert!(tasks.len() >= 2);
        assert!(follow_ups.len() >= 1);
        assert!(reminders.len() >= 1);
    }

    #[test]
    fn test_markdown_checkbox() {
        let detector = ActionDetector::new();
        let text = "- [ ] First task\n- [x] Completed task\n- [ ] Another task";
        let actions = detector.detect(text);

        // Should only detect unchecked boxes
        assert!(actions.len() >= 2);
    }
}
