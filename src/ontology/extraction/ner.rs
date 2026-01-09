//! Named Entity Recognition (NER) for extracting people, organizations, locations, etc.
//!
//! This module provides two extraction approaches:
//! - **Local NER**: Pattern-based extraction using regex and heuristics (fast, offline)
//! - **LLM NER**: LLM-based extraction using external API (more accurate, requires API)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ontology::extraction::LlmExtractionConfig;

// ============================================================================
// Types
// ============================================================================

/// A named entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEntity {
    /// The entity text.
    pub text: String,
    /// Type of entity.
    pub entity_type: NamedEntityType,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Character offset where the entity starts.
    pub start_offset: usize,
    /// Character offset where the entity ends.
    pub end_offset: usize,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// Type of named entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamedEntityType {
    /// A person's name.
    Person,
    /// An organization, company, or team.
    Organization,
    /// A geographic location.
    Location,
    /// An email address.
    Email,
    /// A URL or web address.
    Url,
    /// A phone number.
    PhoneNumber,
    /// A monetary amount.
    Money,
    /// A percentage value.
    Percentage,
    /// A GTD context (@home, @work, etc.).
    Context,
}

// ============================================================================
// NER Extractor Trait
// ============================================================================

/// Trait for NER extractors.
#[async_trait]
pub trait NerExtractor: Send + Sync {
    /// Extract named entities from text.
    async fn extract(&self, text: &str) -> Result<Vec<NamedEntity>>;
}

// ============================================================================
// Local NER Extractor (Pattern-based)
// ============================================================================

/// Pattern-based local NER extractor.
///
/// Uses regex patterns and heuristics to extract entities without external APIs.
/// Suitable for:
/// - Emails
/// - URLs
/// - Phone numbers
/// - Money amounts
/// - Percentages
/// - GTD contexts (@home, @work)
/// - Simple person/organization detection via capitalization
pub struct LocalNerExtractor {
    email_pattern: regex::Regex,
    url_pattern: regex::Regex,
    phone_patterns: Vec<regex::Regex>,
    money_pattern: regex::Regex,
    percentage_pattern: regex::Regex,
    context_pattern: regex::Regex,
    name_pattern: regex::Regex,
    org_indicators: Vec<String>,
    location_indicators: Vec<String>,
}

impl Default for LocalNerExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalNerExtractor {
    /// Create a new local NER extractor with default patterns.
    pub fn new() -> Self {
        Self {
            email_pattern: regex::Regex::new(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            )
            .expect("Invalid email regex"),

            url_pattern: regex::Regex::new(
                r"(?i)\b(?:https?://|www\.)[^\s<>\[\]()]+",
            )
            .expect("Invalid URL regex"),

            phone_patterns: vec![
                // US format
                regex::Regex::new(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
                    .expect("Invalid phone regex"),
                // International format
                regex::Regex::new(r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b")
                    .expect("Invalid phone regex"),
            ],

            money_pattern: regex::Regex::new(
                r"(?:\$|€|£|¥)[\d,]+(?:\.\d{1,2})?|\b(?:USD|EUR|GBP|JPY)\s*[\d,]+(?:\.\d{1,2})?|\b[\d,]+(?:\.\d{1,2})?\s*(?:dollars?|euros?|pounds?|yen)\b",
            )
            .expect("Invalid money regex"),

            percentage_pattern: regex::Regex::new(r"\d+(?:\.\d+)?%")
                .expect("Invalid percentage regex"),

            context_pattern: regex::Regex::new(r"@[a-zA-Z][a-zA-Z0-9_-]*\b")
                .expect("Invalid context regex"),

            // Pattern for potential names (capitalized words)
            name_pattern: regex::Regex::new(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
            )
            .expect("Invalid name regex"),

            org_indicators: vec![
                "Inc".to_string(),
                "Inc.".to_string(),
                "LLC".to_string(),
                "Ltd".to_string(),
                "Ltd.".to_string(),
                "Corp".to_string(),
                "Corp.".to_string(),
                "Corporation".to_string(),
                "Company".to_string(),
                "Co.".to_string(),
                "Co".to_string(),
                "Group".to_string(),
                "Holdings".to_string(),
                "Partners".to_string(),
                "Association".to_string(),
                "Foundation".to_string(),
                "Institute".to_string(),
                "University".to_string(),
                "College".to_string(),
                "School".to_string(),
                "Bank".to_string(),
                "Trust".to_string(),
                "Fund".to_string(),
                "Team".to_string(),
                "Department".to_string(),
                "Dept".to_string(),
                "Division".to_string(),
            ],

            location_indicators: vec![
                "Street".to_string(),
                "St.".to_string(),
                "Avenue".to_string(),
                "Ave.".to_string(),
                "Boulevard".to_string(),
                "Blvd.".to_string(),
                "Road".to_string(),
                "Rd.".to_string(),
                "Drive".to_string(),
                "Dr.".to_string(),
                "Lane".to_string(),
                "Ln.".to_string(),
                "Way".to_string(),
                "Place".to_string(),
                "City".to_string(),
                "Town".to_string(),
                "Village".to_string(),
                "County".to_string(),
                "State".to_string(),
                "Province".to_string(),
                "Country".to_string(),
                "Building".to_string(),
                "Floor".to_string(),
                "Suite".to_string(),
                "Room".to_string(),
            ],
        }
    }

    /// Extract named entities from text using patterns.
    pub fn extract(&self, text: &str) -> Vec<NamedEntity> {
        let mut entities = Vec::new();

        // Extract emails
        for cap in self.email_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            entities.push(NamedEntity {
                text: m.as_str().to_string(),
                entity_type: NamedEntityType::Email,
                confidence: 0.95,
                start_offset: m.start(),
                end_offset: m.end(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Extract URLs
        for cap in self.url_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            // Clean up trailing punctuation
            let url = m.as_str().trim_end_matches(&['.', ',', ')', ']', '!', '?'][..]);
            entities.push(NamedEntity {
                text: url.to_string(),
                entity_type: NamedEntityType::Url,
                confidence: 0.95,
                start_offset: m.start(),
                end_offset: m.start() + url.len(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Extract phone numbers
        for pattern in &self.phone_patterns {
            for cap in pattern.captures_iter(text) {
                let m = cap.get(0).unwrap();
                // Validate it looks like a real phone number (enough digits)
                let digits: String = m.as_str().chars().filter(|c| c.is_ascii_digit()).collect();
                if digits.len() >= 7 {
                    entities.push(NamedEntity {
                        text: m.as_str().to_string(),
                        entity_type: NamedEntityType::PhoneNumber,
                        confidence: 0.85,
                        start_offset: m.start(),
                        end_offset: m.end(),
                        metadata: std::collections::HashMap::new(),
                    });
                }
            }
        }

        // Extract money amounts
        for cap in self.money_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            entities.push(NamedEntity {
                text: m.as_str().to_string(),
                entity_type: NamedEntityType::Money,
                confidence: 0.9,
                start_offset: m.start(),
                end_offset: m.end(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Extract percentages
        for cap in self.percentage_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            entities.push(NamedEntity {
                text: m.as_str().to_string(),
                entity_type: NamedEntityType::Percentage,
                confidence: 0.95,
                start_offset: m.start(),
                end_offset: m.end(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Extract GTD contexts
        for cap in self.context_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            entities.push(NamedEntity {
                text: m.as_str().to_string(),
                entity_type: NamedEntityType::Context,
                confidence: 0.9,
                start_offset: m.start(),
                end_offset: m.end(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Extract names, organizations, and locations using heuristics
        let name_entities = self.extract_names_and_orgs(text);
        entities.extend(name_entities);

        // Sort by offset and remove overlaps
        entities.sort_by_key(|e| e.start_offset);
        Self::remove_overlaps(&mut entities);

        entities
    }

    /// Extract person names and organizations using heuristics.
    fn extract_names_and_orgs(&self, text: &str) -> Vec<NamedEntity> {
        let mut entities = Vec::new();

        // Skip words that are definitely not names
        let skip_words: std::collections::HashSet<&str> = [
            "The", "This", "That", "These", "Those", "What", "Which", "Where", "When", "Why",
            "How", "Who", "Whom", "I", "We", "You", "He", "She", "It", "They", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
            "March", "April", "May", "June", "July", "August", "September", "October", "November",
            "December", "Today", "Tomorrow", "Yesterday", "Next", "Last", "Every", "All", "Some",
            "Any", "Each", "First", "Second", "Third", "New", "Old", "Good", "Bad", "Best",
            "Worst", "Dear", "Hi", "Hello", "Thanks", "Thank", "Please", "Note", "Action",
            "Todo", "Task", "Item", "Important", "Urgent", "Meeting", "Call", "Email", "Phone",
            "But", "And", "For", "With", "From", "About", "After", "Before", "During", "Until",
        ]
        .into_iter()
        .collect();

        for cap in self.name_pattern.captures_iter(text) {
            let m = cap.get(0).unwrap();
            let name = m.as_str();

            // Skip if it's in our skip list
            let first_word = name.split_whitespace().next().unwrap_or("");
            if skip_words.contains(first_word) {
                continue;
            }

            // Skip very short names
            if name.len() < 3 {
                continue;
            }

            // Check for organization indicators
            let is_org = self
                .org_indicators
                .iter()
                .any(|ind| name.ends_with(ind) || name.contains(&format!(" {} ", ind)));

            // Check for location indicators
            let is_location = self.location_indicators.iter().any(|ind| {
                name.ends_with(ind) || name.contains(&format!(" {} ", ind)) || name.contains(ind)
            });

            // Determine entity type
            let entity_type = if is_org {
                NamedEntityType::Organization
            } else if is_location {
                NamedEntityType::Location
            } else {
                // Likely a person name
                // Higher confidence for multi-word names
                NamedEntityType::Person
            };

            // Calculate confidence
            let base_confidence: f32 = if name.contains(' ') {
                0.75 // Multi-word is more likely to be a real name/org
            } else {
                0.6 // Single word is less certain
            };

            // Adjust confidence based on context
            let final_confidence: f32 = if is_org || is_location {
                (base_confidence + 0.1).min(0.9)
            } else {
                base_confidence.min(0.9)
            };

            entities.push(NamedEntity {
                text: name.to_string(),
                entity_type,
                confidence: final_confidence,
                start_offset: m.start(),
                end_offset: m.end(),
                metadata: std::collections::HashMap::new(),
            });
        }

        entities
    }

    /// Remove overlapping entities, keeping the more specific one.
    fn remove_overlaps(entities: &mut Vec<NamedEntity>) {
        if entities.len() < 2 {
            return;
        }

        // Higher priority for more specific types
        fn type_priority(t: NamedEntityType) -> u8 {
            match t {
                NamedEntityType::Email => 10,
                NamedEntityType::Url => 10,
                NamedEntityType::PhoneNumber => 9,
                NamedEntityType::Money => 9,
                NamedEntityType::Percentage => 9,
                NamedEntityType::Context => 8,
                NamedEntityType::Organization => 5,
                NamedEntityType::Location => 5,
                NamedEntityType::Person => 4,
            }
        }

        let mut i = 0;
        while i < entities.len() - 1 {
            let current_end = entities[i].end_offset;
            let next_start = entities[i + 1].start_offset;

            if next_start < current_end {
                // Overlap detected
                let priority_i = type_priority(entities[i].entity_type);
                let priority_j = type_priority(entities[i + 1].entity_type);

                if priority_i >= priority_j {
                    entities.remove(i + 1);
                } else {
                    entities.remove(i);
                }
            } else {
                i += 1;
            }
        }
    }
}

// ============================================================================
// LLM NER Extractor
// ============================================================================

/// LLM-based NER extractor using external API.
///
/// Provides more accurate extraction by leveraging large language models
/// for context-aware entity recognition.
pub struct LlmNerExtractor {
    config: LlmExtractionConfig,
    client: reqwest::Client,
}

impl LlmNerExtractor {
    /// Create a new LLM NER extractor.
    pub fn new(config: LlmExtractionConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Extract named entities using LLM.
    pub async fn extract(&self, text: &str) -> Result<Vec<NamedEntity>> {
        // Truncate text if too long
        let max_chars = self.config.max_tokens * 4; // Rough estimate of chars per token
        let text = if text.len() > max_chars {
            &text[..max_chars]
        } else {
            text
        };

        let prompt = format!(
            r#"Extract named entities from the following text. Return a JSON array of objects with these fields:
- "text": the entity text as it appears
- "type": one of "person", "organization", "location"
- "confidence": a number from 0.0 to 1.0

Only extract clear, unambiguous entities. Focus on:
- Person names (full names preferred)
- Organization/company names
- Geographic locations (cities, countries, addresses)

Text to analyze:
---
{}
---

Return ONLY valid JSON array, no explanation:"#,
            text
        );

        // Get API key from config or environment
        let api_key = self
            .config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());

        let Some(api_key) = api_key else {
            return Err(crate::error::AlloyError::Ontology(
                crate::error::OntologyError::Extraction("LLM NER requires API key".to_string()),
            ));
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.config.api_endpoint))
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a named entity recognition assistant. Extract entities accurately and return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }))
            .send()
            .await
            .map_err(|e| {
                crate::error::AlloyError::Ontology(crate::error::OntologyError::Extraction(
                    format!("LLM API error: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(crate::error::AlloyError::Ontology(
                crate::error::OntologyError::Extraction(format!(
                    "LLM API error {}: {}",
                    status, error_text
                )),
            ));
        }

        let response_json: serde_json::Value = response.json().await.map_err(|e| {
            crate::error::AlloyError::Ontology(crate::error::OntologyError::Extraction(format!(
                "Failed to parse LLM response: {}",
                e
            )))
        })?;

        // Extract content from response
        let content = response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[]");

        // Parse the JSON array
        let extracted: Vec<LlmEntity> = serde_json::from_str(content).unwrap_or_default();

        // Convert to our format and find offsets
        let entities: Vec<NamedEntity> = extracted
            .into_iter()
            .filter_map(|e| {
                let entity_type = match e.entity_type.to_lowercase().as_str() {
                    "person" => NamedEntityType::Person,
                    "organization" | "org" | "company" => NamedEntityType::Organization,
                    "location" | "place" | "geo" => NamedEntityType::Location,
                    _ => return None,
                };

                // Find offset in text
                let start_offset = text.find(&e.text)?;
                let end_offset = start_offset + e.text.len();

                Some(NamedEntity {
                    text: e.text,
                    entity_type,
                    confidence: e.confidence.unwrap_or(0.8),
                    start_offset,
                    end_offset,
                    metadata: std::collections::HashMap::new(),
                })
            })
            .collect();

        Ok(entities)
    }
}

#[derive(Debug, Deserialize)]
struct LlmEntity {
    text: String,
    #[serde(rename = "type")]
    entity_type: String,
    confidence: Option<f32>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_email() {
        let extractor = LocalNerExtractor::new();
        let text = "Contact john.doe@example.com for more info.";
        let entities = extractor.extract(text);

        let emails: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Email)
            .collect();
        assert_eq!(emails.len(), 1);
        assert_eq!(emails[0].text, "john.doe@example.com");
        assert!(emails[0].confidence > 0.9);
    }

    #[test]
    fn test_extract_url() {
        let extractor = LocalNerExtractor::new();
        let text = "Visit https://www.example.com/page for details.";
        let entities = extractor.extract(text);

        let urls: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Url)
            .collect();
        assert_eq!(urls.len(), 1);
        assert!(urls[0].text.contains("example.com"));
    }

    #[test]
    fn test_extract_phone() {
        let extractor = LocalNerExtractor::new();
        let text = "Call us at (555) 123-4567 or +1-555-987-6543.";
        let entities = extractor.extract(text);

        let phones: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::PhoneNumber)
            .collect();
        assert!(phones.len() >= 1);
    }

    #[test]
    fn test_extract_money() {
        let extractor = LocalNerExtractor::new();
        let text = "The project budget is $50,000 and we've spent 10,000 dollars.";
        let entities = extractor.extract(text);

        let money: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Money)
            .collect();
        assert!(!money.is_empty(), "Should detect money amounts");
    }

    #[test]
    fn test_extract_percentage() {
        let extractor = LocalNerExtractor::new();
        let text = "We achieved 95% completion rate and 25% growth.";
        let entities = extractor.extract(text);

        let percentages: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Percentage)
            .collect();
        assert!(!percentages.is_empty(), "Should detect percentage values");
    }

    #[test]
    fn test_extract_context() {
        let extractor = LocalNerExtractor::new();
        let text = "Do this @home and that @work. Also check @computer tasks.";
        let entities = extractor.extract(text);

        let contexts: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Context)
            .collect();
        assert_eq!(contexts.len(), 3);
        assert!(contexts.iter().any(|c| c.text == "@home"));
        assert!(contexts.iter().any(|c| c.text == "@work"));
    }

    #[test]
    fn test_extract_person() {
        let extractor = LocalNerExtractor::new();
        let text = "Meeting with John Smith and Sarah Johnson tomorrow.";
        let entities = extractor.extract(text);

        let people: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Person)
            .collect();
        // Should find at least John Smith and Sarah Johnson
        assert!(people.len() >= 2);
    }

    #[test]
    fn test_extract_organization() {
        let extractor = LocalNerExtractor::new();
        let text = "The proposal was sent to Acme Corp and BigTech Inc.";
        let entities = extractor.extract(text);

        let orgs: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Organization)
            .collect();
        assert!(orgs.len() >= 1);
    }

    #[test]
    fn test_extract_location() {
        let extractor = LocalNerExtractor::new();
        let text = "The office is at 123 Main Street in New York City.";
        let entities = extractor.extract(text);

        let locations: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == NamedEntityType::Location)
            .collect();
        assert!(locations.len() >= 1);
    }

    #[test]
    fn test_no_overlap() {
        let extractor = LocalNerExtractor::new();
        let text = "Email john@example.com for John Smith at Acme Corp.";
        let entities = extractor.extract(text);

        // Check no overlapping entities
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                assert!(
                    entities[i].end_offset <= entities[j].start_offset
                        || entities[j].end_offset <= entities[i].start_offset,
                    "Entities overlap: {:?} and {:?}",
                    entities[i],
                    entities[j]
                );
            }
        }
    }

    #[test]
    fn test_skip_common_words() {
        let extractor = LocalNerExtractor::new();
        let text = "The meeting is on Monday. This is important.";
        let entities = extractor.extract(text);

        // Should not extract "The", "Monday", "This" as entities
        let false_positives: Vec<_> = entities
            .iter()
            .filter(|e| ["The", "Monday", "This"].contains(&e.text.as_str()))
            .collect();
        assert!(false_positives.is_empty());
    }

    #[test]
    fn test_complex_text() {
        let extractor = LocalNerExtractor::new();
        let text = r#"
From: John Smith <john.smith@acme.com>
To: Sarah Johnson <sarah@example.org>

Hi Sarah,

Please review the proposal. The total cost is $150,000.
Our target completion rate is 98%. Call me at (555) 123-4567.

Tasks:
- @home: Review documents
- @work: Prepare presentation

Visit https://www.example.com/proposal for details.

Thanks,
John
        "#;

        let entities = extractor.extract(text);

        // Should find various entity types
        assert!(entities.iter().any(|e| e.entity_type == NamedEntityType::Email), "Should find emails");
        assert!(entities.iter().any(|e| e.entity_type == NamedEntityType::Context), "Should find contexts");
        assert!(entities.iter().any(|e| e.entity_type == NamedEntityType::Url), "Should find URLs");
    }
}
