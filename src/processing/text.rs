//! Text processor for plain text, markdown, JSON, YAML, and code files.

use async_trait::async_trait;
use bytes::Bytes;
use tracing::debug;

use crate::error::Result;
use crate::processing::{ContentMetadata, ProcessedContent, Processor};
use crate::sources::SourceItem;

/// Processor for text-based files.
pub struct TextProcessor {
    supported_types: Vec<&'static str>,
}

impl TextProcessor {
    /// Create a new text processor.
    pub fn new() -> Self {
        Self {
            supported_types: vec![
                // Plain text
                "text/plain",
                "text/*",
                // Markup
                "text/markdown",
                "text/x-markdown",
                "text/html",
                "text/xml",
                // Data formats
                "application/json",
                "application/x-yaml",
                "application/yaml",
                "text/yaml",
                "text/x-yaml",
                "application/toml",
                "text/toml",
                "application/xml",
                // Code files
                "text/x-python",
                "text/x-rust",
                "text/x-go",
                "text/javascript",
                "application/javascript",
                "text/typescript",
                "application/typescript",
                "text/x-c",
                "text/x-c++",
                "text/x-java",
                "text/x-ruby",
                "text/x-perl",
                "text/x-php",
                "text/x-shellscript",
                "application/x-sh",
                // Config files
                "text/csv",
                "text/tab-separated-values",
            ],
        }
    }
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Processor for TextProcessor {
    async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent> {
        // Decode content as UTF-8 (with lossy replacement for invalid sequences)
        let text = String::from_utf8_lossy(&content).to_string();

        debug!(
            uri = %item.uri,
            mime_type = %item.mime_type,
            text_len = text.len(),
            "Processed text file"
        );

        // Extract metadata based on content type
        let metadata = extract_metadata(&text, &item.mime_type);

        Ok(ProcessedContent::with_metadata(text, metadata))
    }

    fn supported_types(&self) -> &[&str] {
        &self.supported_types
    }

    fn name(&self) -> &str {
        "text"
    }
}

/// Extract metadata from text content based on MIME type.
fn extract_metadata(text: &str, mime_type: &str) -> ContentMetadata {
    let mut metadata = ContentMetadata::from_text(text);

    // Try to extract title from markdown
    if mime_type.contains("markdown") {
        metadata.title = extract_markdown_title(text);
    }

    // Try to extract info from JSON
    if mime_type.contains("json") {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
            if let Some(title) = json.get("title").and_then(|v| v.as_str()) {
                metadata.title = Some(title.to_string());
            }
            if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
                metadata.title = metadata.title.or_else(|| Some(name.to_string()));
            }
        }
    }

    // Try to extract info from YAML frontmatter
    if mime_type.contains("markdown") || mime_type.contains("yaml") {
        if let Some(fm) = extract_yaml_frontmatter(text) {
            if let Ok(yaml) = serde_json::from_str::<serde_json::Value>(&fm) {
                if let Some(title) = yaml.get("title").and_then(|v| v.as_str()) {
                    metadata.title = Some(title.to_string());
                }
                if let Some(author) = yaml.get("author").and_then(|v| v.as_str()) {
                    metadata.author = Some(author.to_string());
                }
            }
        }
    }

    metadata
}

/// Extract title from markdown content (first H1 heading).
fn extract_markdown_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            return Some(title.trim().to_string());
        }
    }
    None
}

/// Extract YAML frontmatter from text.
fn extract_yaml_frontmatter(text: &str) -> Option<String> {
    if !text.starts_with("---") {
        return None;
    }

    let rest = &text[3..];
    if let Some(end) = rest.find("\n---") {
        let yaml = rest[..end].trim();
        // Convert YAML to JSON for parsing
        // Simple approach: just extract key-value pairs
        let mut json_obj = serde_json::Map::new();
        for line in yaml.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim().trim_matches('"').trim_matches('\'');
                json_obj.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
        return Some(serde_json::to_string(&json_obj).unwrap_or_default());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_item(mime_type: &str) -> SourceItem {
        SourceItem {
            id: "test".to_string(),
            uri: "/test/file.txt".to_string(),
            mime_type: mime_type.to_string(),
            size: 100,
            modified: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn test_process_plain_text() {
        let processor = TextProcessor::new();
        let content = Bytes::from("Hello, world!");
        let item = make_item("text/plain");

        let result = processor.process(content, &item).await.unwrap();
        assert_eq!(result.text, "Hello, world!");
        assert_eq!(result.metadata.word_count, Some(2));
        assert_eq!(result.metadata.char_count, Some(13));
    }

    #[tokio::test]
    async fn test_process_markdown() {
        let processor = TextProcessor::new();
        let content = Bytes::from("# My Title\n\nSome content here.");
        let item = make_item("text/markdown");

        let result = processor.process(content, &item).await.unwrap();
        assert_eq!(result.metadata.title, Some("My Title".to_string()));
    }

    #[tokio::test]
    async fn test_process_json() {
        let processor = TextProcessor::new();
        let content = Bytes::from(r#"{"title": "Test Document", "data": [1, 2, 3]}"#);
        let item = make_item("application/json");

        let result = processor.process(content, &item).await.unwrap();
        assert_eq!(result.metadata.title, Some("Test Document".to_string()));
    }

    #[tokio::test]
    async fn test_process_yaml_frontmatter() {
        let processor = TextProcessor::new();
        let content = Bytes::from("---\ntitle: My Doc\nauthor: John\n---\n\nContent here.");
        let item = make_item("text/markdown");

        let result = processor.process(content, &item).await.unwrap();
        assert_eq!(result.metadata.title, Some("My Doc".to_string()));
        assert_eq!(result.metadata.author, Some("John".to_string()));
    }

    #[test]
    fn test_supports_types() {
        let processor = TextProcessor::new();
        assert!(processor.supports("text/plain"));
        assert!(processor.supports("application/json"));
        assert!(processor.supports("text/markdown"));
        assert!(!processor.supports("application/pdf"));
    }

    #[test]
    fn test_extract_markdown_title() {
        assert_eq!(
            extract_markdown_title("# Hello World\n\nContent"),
            Some("Hello World".to_string())
        );
        assert_eq!(extract_markdown_title("No title here"), None);
        assert_eq!(
            extract_markdown_title("## Not H1\n# Actual Title"),
            Some("Actual Title".to_string())
        );
    }

    #[test]
    fn test_extract_yaml_frontmatter() {
        let text = "---\ntitle: Test\nauthor: Me\n---\nContent";
        let fm = extract_yaml_frontmatter(text).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&fm).unwrap();
        assert_eq!(parsed["title"], "Test");
        assert_eq!(parsed["author"], "Me");
    }

    #[test]
    fn test_no_yaml_frontmatter() {
        assert!(extract_yaml_frontmatter("No frontmatter here").is_none());
    }
}
