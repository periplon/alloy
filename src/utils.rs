//! Utility functions for string manipulation and other common operations.

/// Find the nearest valid UTF-8 char boundary at or before the given byte index.
#[inline]
pub fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Truncate a string to approximately `max_len` bytes, ensuring valid UTF-8 boundaries.
/// Returns a slice of the original string.
#[inline]
pub fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let boundary = floor_char_boundary(s, max_len);
        &s[..boundary]
    }
}

/// Safely slice a string between two byte positions, adjusting to valid UTF-8 boundaries.
/// The start is adjusted forward (ceil), the end is adjusted backward (floor).
#[inline]
pub fn safe_slice(s: &str, start: usize, end: usize) -> &str {
    let safe_start = ceil_char_boundary(s, start);
    let safe_end = floor_char_boundary(s, end);
    if safe_start >= safe_end {
        ""
    } else {
        &s[safe_start..safe_end]
    }
}

/// Find the nearest valid UTF-8 char boundary at or after the given byte index.
#[inline]
pub fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_char_boundary_ascii() {
        let s = "hello";
        assert_eq!(floor_char_boundary(s, 0), 0);
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 5), 5);
        assert_eq!(floor_char_boundary(s, 10), 5);
    }

    #[test]
    fn test_floor_char_boundary_utf8() {
        // '─' is 3 bytes (E2 94 80)
        let s = "a─b";
        assert_eq!(floor_char_boundary(s, 0), 0); // 'a'
        assert_eq!(floor_char_boundary(s, 1), 1); // start of '─'
        assert_eq!(floor_char_boundary(s, 2), 1); // middle of '─', goes back to 1
        assert_eq!(floor_char_boundary(s, 3), 1); // middle of '─', goes back to 1
        assert_eq!(floor_char_boundary(s, 4), 4); // 'b'
    }

    #[test]
    fn test_truncate_str() {
        let s = "hello─world";
        // '─' starts at byte 5, is 3 bytes
        assert_eq!(truncate_str(s, 5), "hello");
        assert_eq!(truncate_str(s, 6), "hello"); // mid-char, truncates to 5
        assert_eq!(truncate_str(s, 7), "hello"); // mid-char, truncates to 5
        assert_eq!(truncate_str(s, 8), "hello─");
        assert_eq!(truncate_str(s, 100), s);
    }

    #[test]
    fn test_safe_slice() {
        let s = "hello─world";
        // '─' is at bytes 5..8, "world" is at bytes 8..13
        assert_eq!(safe_slice(s, 0, 5), "hello");
        assert_eq!(safe_slice(s, 0, 6), "hello"); // end mid-char, adjusted to 5
        assert_eq!(safe_slice(s, 6, 11), "wor"); // start mid-char (6->8), end at 11
        assert_eq!(safe_slice(s, 6, 13), "world"); // start mid-char (6->8), end at 13
        assert_eq!(safe_slice(s, 5, 8), "─");
    }
}
