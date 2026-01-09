//! Text chunking utilities.

use crate::processing::{ChunkConfig, TextChunk};

/// Split text into chunks based on configuration.
pub fn chunk_text(text: &str, doc_id: &str, config: &ChunkConfig) -> Vec<TextChunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let total_len = chars.len();

    if total_len <= config.chunk_size {
        // Text fits in a single chunk
        chunks.push(TextChunk::new(
            format!("{}-0", doc_id),
            0,
            text.to_string(),
            0,
            total_len,
        ));
        return chunks;
    }

    let mut start = 0;
    let mut chunk_index = 0;

    while start < total_len {
        let mut end = (start + config.chunk_size).min(total_len);

        // If not at the end of text, try to find a good break point
        if end < total_len && config.respect_sentences {
            end = find_break_point(&chars, start, end, config);
        }

        // Extract chunk text
        let chunk_text: String = chars[start..end].iter().collect();

        // Skip empty or whitespace-only chunks
        if !chunk_text.trim().is_empty() {
            chunks.push(TextChunk::new(
                format!("{}-{}", doc_id, chunk_index),
                chunk_index,
                chunk_text,
                start,
                end,
            ));
            chunk_index += 1;
        }

        // Calculate next start with overlap
        let step = if end - start > config.chunk_overlap {
            end - start - config.chunk_overlap
        } else {
            end - start
        };

        start += step;

        // Safety check to prevent infinite loops
        if step == 0 {
            break;
        }
    }

    // Merge small final chunk if needed
    if chunks.len() >= 2 {
        let last_idx = chunks.len() - 1;
        if chunks[last_idx].text.len() < config.min_chunk_size {
            let last = chunks.pop().unwrap();
            if let Some(prev) = chunks.last_mut() {
                prev.text.push_str(&last.text);
                prev.end_offset = last.end_offset;
            }
        }
    }

    chunks
}

/// Find a good break point (sentence or paragraph boundary).
fn find_break_point(chars: &[char], start: usize, target_end: usize, config: &ChunkConfig) -> usize {
    let search_start = if target_end > start + 50 {
        target_end - 50
    } else {
        start
    };

    // First try to find paragraph break
    if config.respect_paragraphs {
        for i in (search_start..target_end).rev() {
            if i + 1 < chars.len() && chars[i] == '\n' && chars[i + 1] == '\n' {
                return i + 2;
            }
        }
    }

    // Then try sentence boundaries
    if config.respect_sentences {
        for i in (search_start..target_end).rev() {
            if is_sentence_end(chars, i) {
                return i + 1;
            }
        }
    }

    // Fall back to word boundary
    for i in (search_start..target_end).rev() {
        if chars[i].is_whitespace() {
            return i + 1;
        }
    }

    target_end
}

/// Check if position is a sentence end.
fn is_sentence_end(chars: &[char], pos: usize) -> bool {
    let c = chars[pos];
    if c == '.' || c == '!' || c == '?' {
        // Check if followed by whitespace or end
        if pos + 1 >= chars.len() {
            return true;
        }
        let next = chars[pos + 1];
        return next.is_whitespace();
    }
    false
}

/// Split text by paragraphs first, then chunk each paragraph.
pub fn chunk_by_paragraphs(text: &str, doc_id: &str, config: &ChunkConfig) -> Vec<TextChunk> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut all_chunks = Vec::new();
    let mut global_offset = 0;
    let mut chunk_index = 0;

    for para in paragraphs {
        if para.trim().is_empty() {
            global_offset += para.len() + 2; // +2 for \n\n
            continue;
        }

        let para_chunks = chunk_text(para, &format!("{}-p{}", doc_id, chunk_index), config);

        for mut chunk in para_chunks {
            chunk.start_offset += global_offset;
            chunk.end_offset += global_offset;
            chunk.index = chunk_index;
            chunk.id = format!("{}-{}", doc_id, chunk_index);
            all_chunks.push(chunk);
            chunk_index += 1;
        }

        global_offset += para.len() + 2;
    }

    all_chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ChunkConfig {
        ChunkConfig {
            chunk_size: 100,
            chunk_overlap: 20,
            min_chunk_size: 20,
            respect_sentences: true,
            respect_paragraphs: true,
        }
    }

    #[test]
    fn test_chunk_short_text() {
        let config = default_config();
        let chunks = chunk_text("Short text.", "doc1", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short text.");
    }

    #[test]
    fn test_chunk_empty_text() {
        let config = default_config();
        let chunks = chunk_text("", "doc1", &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_long_text() {
        let config = ChunkConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            min_chunk_size: 10,
            respect_sentences: true,
            respect_paragraphs: true,
        };

        let text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four.";
        let chunks = chunk_text(text, "doc1", &config);

        assert!(chunks.len() >= 2);
        // Verify overlap exists
        for i in 1..chunks.len() {
            // Each chunk should start before the previous one ends (overlap)
            assert!(chunks[i].start_offset < chunks[i - 1].end_offset);
        }
    }

    #[test]
    fn test_chunk_respects_sentences() {
        let config = ChunkConfig {
            chunk_size: 30,
            chunk_overlap: 5,
            min_chunk_size: 5,
            respect_sentences: true,
            respect_paragraphs: false,
        };

        let text = "Hello world. Goodbye world.";
        let chunks = chunk_text(text, "doc1", &config);

        // Should break at sentence boundary
        assert!(chunks.len() >= 1);
        // First chunk should end after a sentence
        let first_chunk = &chunks[0].text;
        assert!(first_chunk.ends_with('.') || first_chunk.ends_with(". "));
    }

    #[test]
    fn test_chunk_by_paragraphs() {
        let config = ChunkConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            min_chunk_size: 10,
            respect_sentences: true,
            respect_paragraphs: true,
        };

        let text = "First paragraph here.\n\nSecond paragraph here.";
        let chunks = chunk_by_paragraphs(text, "doc1", &config);

        assert!(!chunks.is_empty());
        // Chunks should maintain sequential indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn test_is_sentence_end() {
        let chars: Vec<char> = "Hello. World".chars().collect();
        assert!(is_sentence_end(&chars, 5)); // Period followed by space

        let chars: Vec<char> = "Hello.World".chars().collect();
        assert!(!is_sentence_end(&chars, 5)); // Period not followed by space
    }
}
