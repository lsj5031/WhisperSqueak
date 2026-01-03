"""
Tests for SSE parsing using the actual sse_parser module.

These tests verify the SSE event parsing logic handles:
- Standard JSON format: data: {"data": "text"}
- Raw text format: data: text (for glm-asr backend)
- Completion sentinel: data: [DONE]
- Error sentinel: data: [Error: message]
- Incomplete/malformed events
- Buffer handling across chunk boundaries
"""

import sys
from pathlib import Path

# Add parent directory to path for importing sse_parser
sys.path.insert(0, str(Path(__file__).parent.parent))

from sse_parser import parse_sse_line, SSEParser, SSEParseResult

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class TestSSELineParsing:
    """Test individual SSE line parsing."""
    
    def test_empty_line(self):
        """Empty lines should be ignored."""
        result = parse_sse_line("", "existing")
        assert result.accumulated_text == "existing"
        assert result.is_done is False
        assert result.error is None
    
    def test_non_data_line(self):
        """Non-data lines (comments, etc) should be ignored."""
        result = parse_sse_line(": comment", "existing")
        assert result.accumulated_text == "existing"
        assert result.is_done is False
        assert result.error is None
    
    def test_json_format(self):
        """Standard JSON format should be parsed correctly."""
        result = parse_sse_line('data: {"data": "Hello "}', "")
        assert result.accumulated_text == "Hello "
        assert result.is_done is False
        assert result.error is None
    
    def test_json_format_accumulation(self):
        """JSON data should accumulate with existing text."""
        result = parse_sse_line('data: {"data": "world"}', "Hello ")
        assert result.accumulated_text == "Hello world"
        assert result.is_done is False
        assert result.error is None
    
    def test_raw_text_format(self):
        """Raw text (non-JSON) should be used as-is."""
        result = parse_sse_line("data: Hello world", "")
        assert result.accumulated_text == "Hello world"
        assert result.is_done is False
        assert result.error is None
    
    def test_raw_text_accumulation(self):
        """Raw text should accumulate."""
        result = parse_sse_line("data: world", "Hello ")
        assert result.accumulated_text == "Hello world"
        assert result.is_done is False
        assert result.error is None
    
    def test_done_sentinel(self):
        """[DONE] sentinel should signal completion."""
        result = parse_sse_line("data: [DONE]", "Hello")
        assert result.accumulated_text == "Hello"
        assert result.is_done is True
        assert result.error is None
    
    def test_error_sentinel_with_bracket(self):
        """[Error: message] should extract error message."""
        result = parse_sse_line("data: [Error: Model not found]", "")
        assert result.accumulated_text == ""
        assert result.is_done is False
        assert result.error == "Model not found"
    
    def test_error_sentinel_without_bracket(self):
        """[Error: message (malformed) should still extract."""
        result = parse_sse_line("data: [Error: Connection lost", "")
        assert result.accumulated_text == ""
        assert result.is_done is False
        assert result.error == "Connection lost"
    
    def test_empty_json_data(self):
        """Empty data field in JSON should not accumulate."""
        result = parse_sse_line('data: {"data": ""}', "existing")
        assert result.accumulated_text == "existing"
        assert result.is_done is False
        assert result.error is None
    
    def test_json_with_other_fields(self):
        """JSON with extra fields should still extract data."""
        result = parse_sse_line(
            'data: {"data": "text", "timestamp": 123, "confidence": 0.95}', ""
        )
        assert result.accumulated_text == "text"
        assert result.is_done is False
        assert result.error is None
    
    def test_whitespace_handling(self):
        """Extra whitespace around data should be handled."""
        result = parse_sse_line("data:   Hello   ", "")
        assert result.accumulated_text == "Hello"  # Note: strip happens on data_str
        assert result.is_done is False
        assert result.error is None
    
    def test_unicode_text(self):
        """Unicode characters should be preserved."""
        result = parse_sse_line('data: {"data": "你好世界 🌍"}', "")
        assert result.accumulated_text == "你好世界 🌍"
        assert result.is_done is False
        assert result.error is None
    
    def test_multiline_json_value(self):
        """JSON with escaped newlines should parse correctly."""
        result = parse_sse_line('data: {"data": "line1\\nline2"}', "")
        assert result.accumulated_text == "line1\nline2"
        assert result.is_done is False
        assert result.error is None


class TestSSEParser:
    """Test the stateful SSEParser class."""
    
    def test_complete_events(self):
        """Multiple complete events should all be parsed."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "Hello "}\ndata: {"data": "world"}\n')
        
        assert len(results) == 2
        assert results[0].accumulated_text == "Hello "
        assert results[1].accumulated_text == "Hello world"
    
    def test_incomplete_event(self):
        """Incomplete event at end should stay in buffer."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "Hello "}\ndata: {"data": "wor')
        
        assert len(results) == 1
        assert results[0].accumulated_text == "Hello "
        
        # Feed more data
        results = parser.feed('ld"}\n')
        assert len(results) == 1
        assert results[0].accumulated_text == "Hello world"
    
    def test_done_in_stream(self):
        """[DONE] should be detected and stop processing."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "text"}\ndata: [DONE]\n')
        
        assert len(results) == 2
        assert results[0].accumulated_text == "text"
        assert results[1].is_done is True
        assert parser.get_text() == "text"
    
    def test_finalize(self):
        """Finalize should process remaining buffer."""
        parser = SSEParser()
        parser.feed('data: {"data": "Hello"}\ndata: {"data": " world"}')
        
        # First event processed, second in buffer
        final = parser.finalize()
        assert final is not None
        assert parser.get_text() == "Hello world"
    
    def test_mixed_formats(self):
        """Mix of JSON and raw text should both work."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "Hello "}\ndata: world\n')
        
        assert len(results) == 2
        assert results[0].accumulated_text == "Hello "
        assert results[1].accumulated_text == "Hello world"
    
    def test_empty_lines_skipped(self):
        """Empty lines between events should be ignored."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "a"}\n\n\ndata: {"data": "b"}\n')
        
        # Only data lines produce results
        assert len(results) == 2
        assert results[1].accumulated_text == "ab"
    
    def test_error_stops_processing(self):
        """Error sentinel should stop processing."""
        parser = SSEParser()
        results = parser.feed('data: {"data": "text"}\ndata: [Error: failed]\ndata: more\n')
        
        # Should have text event and error event, but not the "more" event
        assert len(results) == 2
        assert results[0].accumulated_text == "text"
        assert results[1].error == "failed"


class TestEdgeCases:
    """Edge cases and potential failure modes."""
    
    def test_very_long_text(self):
        """Very long text should be handled."""
        long_text = "x" * 10000
        result = parse_sse_line(f'data: {{"data": "{long_text}"}}', "")
        assert result.accumulated_text == long_text
    
    def test_special_json_characters(self):
        """Special characters in JSON should be escaped properly."""
        result = parse_sse_line(
            r'data: {"data": "He said \"hello\""}', ""
        )
        assert result.accumulated_text == 'He said "hello"'
    
    def test_json_with_nested_objects(self):
        """Nested objects should not break parsing."""
        # We only care about top-level "data" field
        result = parse_sse_line(
            'data: {"data": "text", "metadata": {"nested": true}}', ""
        )
        assert result.accumulated_text == "text"
    
    def test_malformed_json_fallback(self):
        """Invalid JSON should fall back to raw text."""
        result = parse_sse_line('data: {not valid json}', "")
        assert result.accumulated_text == "{not valid json}"
        assert result.error is None
    
    def test_data_prefix_in_content(self):
        """'data:' appearing in content should not confuse parser."""
        result = parse_sse_line('data: The URL is data://example.com', "")
        assert result.accumulated_text == "The URL is data://example.com"
    
    def test_colon_without_space(self):
        """'data:value' (no space) should still work per SSE spec."""
        result = parse_sse_line('data:Hello', "")
        assert result.accumulated_text == "Hello"
    
    def test_only_data_prefix(self):
        """Just 'data:' with nothing after should not crash."""
        result = parse_sse_line('data:', "existing")
        assert result.accumulated_text == "existing"  # Empty data_str, nothing to add
    
    def test_just_whitespace_data(self):
        """'data:   ' (only whitespace) should result in empty."""
        result = parse_sse_line('data:   ', "existing")
        assert result.accumulated_text == "existing"  # Stripped to empty


def run_tests_without_pytest():
    """Run tests manually when pytest is not available."""
    test_classes = [TestSSELineParsing, TestSSEParser, TestEdgeCases]
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return 0 if failed == 0 else 1


# Run with: pytest tests/test_sse_parsing.py -v
# Or directly: python tests/test_sse_parsing.py
if __name__ == "__main__":
    if HAS_PYTEST:
        import pytest
        pytest.main([__file__, "-v"])
    else:
        exit(run_tests_without_pytest())
