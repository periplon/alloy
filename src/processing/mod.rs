//! Processing module for document text extraction and chunking.
//!
//! This module provides processors for various document types:
//! - Text: Plain text, Markdown, JSON, YAML, and code files
//! - Documents: PDF and DOCX files
//! - Images: With OCR, CLIP embeddings, and Vision API support
//!
//! The `ProcessorRegistry` provides a unified interface to process any supported file type.

mod chunker;
mod document;
mod image;
mod registry;
mod text;
mod traits;

pub use chunker::{chunk_by_paragraphs, chunk_text};
pub use document::{DocxProcessor, PdfProcessor};
pub use image::{ClipEmbedder, ImageProcessor, VisionApiClient};
pub use registry::ProcessorRegistry;
pub use text::TextProcessor;
pub use traits::*;
