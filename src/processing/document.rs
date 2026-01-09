//! Document processors for PDF and DOCX files.

use async_trait::async_trait;
use bytes::Bytes;
use tracing::{debug, warn};

use crate::error::{ProcessingError, Result};
use crate::processing::{ContentMetadata, ProcessedContent, Processor};
use crate::sources::SourceItem;

/// Processor for PDF files.
pub struct PdfProcessor {
    supported_types: Vec<&'static str>,
}

impl PdfProcessor {
    /// Create a new PDF processor.
    pub fn new() -> Self {
        Self {
            supported_types: vec!["application/pdf"],
        }
    }
}

impl Default for PdfProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Processor for PdfProcessor {
    async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent> {
        // Use pdf-extract to extract text
        let text = match pdf_extract::extract_text_from_mem(&content) {
            Ok(text) => text,
            Err(e) => {
                warn!(uri = %item.uri, error = %e, "Failed to extract PDF text");
                return Err(ProcessingError::Pdf(e.to_string()).into());
            }
        };

        // Clean up extracted text
        let text = clean_pdf_text(&text);

        // Extract metadata
        let mut metadata = ContentMetadata::from_text(&text);
        metadata.extra = serde_json::json!({
            "format": "pdf",
        });

        debug!(
            uri = %item.uri,
            text_len = text.len(),
            word_count = metadata.word_count,
            "Processed PDF"
        );

        Ok(ProcessedContent::with_metadata(text, metadata))
    }

    fn supported_types(&self) -> &[&str] {
        &self.supported_types
    }

    fn name(&self) -> &str {
        "pdf"
    }
}

/// Clean up PDF extracted text.
fn clean_pdf_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_whitespace = false;

    for c in text.chars() {
        if c.is_whitespace() {
            if c == '\n' {
                // Preserve paragraph breaks (double newlines)
                if !last_was_whitespace {
                    result.push('\n');
                }
            } else if !last_was_whitespace {
                result.push(' ');
            }
            last_was_whitespace = true;
        } else {
            result.push(c);
            last_was_whitespace = false;
        }
    }

    result.trim().to_string()
}

/// Processor for DOCX files.
pub struct DocxProcessor {
    supported_types: Vec<&'static str>,
}

impl DocxProcessor {
    /// Create a new DOCX processor.
    pub fn new() -> Self {
        Self {
            supported_types: vec![
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ],
        }
    }
}

impl Default for DocxProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Processor for DocxProcessor {
    async fn process(&self, content: Bytes, item: &SourceItem) -> Result<ProcessedContent> {
        // Use docx-rs to extract text
        let docx = match docx_rs::read_docx(&content) {
            Ok(doc) => doc,
            Err(e) => {
                warn!(uri = %item.uri, error = %e, "Failed to read DOCX");
                return Err(ProcessingError::Docx(e.to_string()).into());
            }
        };

        // Extract text from document
        let text = extract_docx_text(&docx);

        // Extract metadata
        let mut metadata = ContentMetadata::from_text(&text);
        metadata.extra = serde_json::json!({
            "format": "docx",
        });

        debug!(
            uri = %item.uri,
            text_len = text.len(),
            word_count = metadata.word_count,
            "Processed DOCX"
        );

        Ok(ProcessedContent::with_metadata(text, metadata))
    }

    fn supported_types(&self) -> &[&str] {
        &self.supported_types
    }

    fn name(&self) -> &str {
        "docx"
    }
}

/// Extract text content from a DOCX document.
fn extract_docx_text(docx: &docx_rs::Docx) -> String {
    let mut text = String::new();

    for child in docx.document.children.iter() {
        extract_document_child(&mut text, child);
    }

    text.trim().to_string()
}

/// Recursively extract text from document children.
fn extract_document_child(text: &mut String, child: &docx_rs::DocumentChild) {
    match child {
        docx_rs::DocumentChild::Paragraph(para) => {
            let para_text = extract_paragraph_text(para);
            if !para_text.is_empty() {
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&para_text);
            }
        }
        docx_rs::DocumentChild::Table(table) => {
            for row in &table.rows {
                match row {
                    docx_rs::TableChild::TableRow(tr) => {
                        let mut row_texts = Vec::new();
                        for cell in &tr.cells {
                            match cell {
                                docx_rs::TableRowChild::TableCell(tc) => {
                                    let mut cell_text = String::new();
                                    for child in &tc.children {
                                        if let docx_rs::TableCellContent::Paragraph(p) = child {
                                            let pt = extract_paragraph_text(p);
                                            if !pt.is_empty() {
                                                if !cell_text.is_empty() {
                                                    cell_text.push(' ');
                                                }
                                                cell_text.push_str(&pt);
                                            }
                                        }
                                    }
                                    if !cell_text.is_empty() {
                                        row_texts.push(cell_text);
                                    }
                                }
                            }
                        }
                        if !row_texts.is_empty() {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&row_texts.join(" | "));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Extract text from a paragraph.
fn extract_paragraph_text(para: &docx_rs::Paragraph) -> String {
    let mut text = String::new();

    for child in &para.children {
        match child {
            docx_rs::ParagraphChild::Run(run) => {
                for child in &run.children {
                    if let docx_rs::RunChild::Text(t) = child {
                        text.push_str(&t.text);
                    }
                }
            }
            docx_rs::ParagraphChild::Hyperlink(link) => {
                for child in &link.children {
                    if let docx_rs::ParagraphChild::Run(run) = child {
                        for child in &run.children {
                            if let docx_rs::RunChild::Text(t) = child {
                                text.push_str(&t.text);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_processor_supports() {
        let processor = PdfProcessor::new();
        assert!(processor.supports("application/pdf"));
        assert!(!processor.supports("text/plain"));
    }

    #[test]
    fn test_docx_processor_supports() {
        let processor = DocxProcessor::new();
        assert!(processor
            .supports("application/vnd.openxmlformats-officedocument.wordprocessingml.document"));
        assert!(!processor.supports("text/plain"));
    }

    #[test]
    fn test_clean_pdf_text() {
        let messy = "Hello   world\n\n\n  multiple   spaces";
        let clean = clean_pdf_text(messy);
        assert!(!clean.contains("   "));
    }

    #[test]
    fn test_clean_pdf_preserves_content() {
        let text = "Hello world. This is a test.";
        let cleaned = clean_pdf_text(text);
        assert!(cleaned.contains("Hello"));
        assert!(cleaned.contains("world"));
        assert!(cleaned.contains("test"));
    }
}
