"""
SSE (Server-Sent Events) parser for Whisper transcription streams.

This module provides reusable SSE parsing logic for extracting transcription
text from streaming responses. It handles both JSON and raw text formats.

Expected SSE Protocol:
----------------------
- Progress updates: `data: {"data": "transcribed text segment"}`
  OR raw text:      `data: transcribed text segment`
- Completion:       `data: [DONE]`
- Errors:           `data: [Error: error message here]`
"""

import json
from dataclasses import dataclass


@dataclass
class SSEParseResult:
    """Result of parsing a single SSE line."""

    accumulated_text: str
    is_done: bool
    error: str | None


def parse_sse_line(line: str, accumulated_text: str) -> SSEParseResult:
    """
    Parse a single SSE line and return updated state.

    Args:
        line: The SSE line to parse (already stripped)
        accumulated_text: Current accumulated transcription text

    Returns:
        SSEParseResult with updated accumulated_text, done flag, and any error
    """
    if not line or not line.startswith("data:"):
        return SSEParseResult(accumulated_text, False, None)

    # Extract data value (handle both "data: value" and "data:value")
    data_str = line[5:].strip()

    # Check for completion sentinel
    if data_str == "[DONE]":
        return SSEParseResult(accumulated_text, True, None)

    # Check for error sentinel
    if data_str.startswith("[Error:"):
        error_start = len("[Error:")
        error_msg = data_str[error_start:]
        if error_msg.endswith("]"):
            error_msg = error_msg[:-1]
        return SSEParseResult(accumulated_text, False, error_msg.strip())

    # Empty data after stripping - nothing to add
    if not data_str:
        return SSEParseResult(accumulated_text, False, None)

    # Parse JSON data
    try:
        data = json.loads(data_str)
        text = data.get("data", "")
        if text:
            accumulated_text += text
    except json.JSONDecodeError:
        # If not JSON, treat raw data as text
        accumulated_text += data_str

    return SSEParseResult(accumulated_text, False, None)


class SSEParser:
    """
    Stateful SSE parser that processes streaming chunks.

    Maintains an internal buffer to handle events split across chunks.
    """

    def __init__(self):
        self.buffer = ""
        self.accumulated_text = ""

    def feed(self, chunk: str) -> list[SSEParseResult]:
        """
        Feed a chunk of SSE data and return any complete parse results.

        Args:
            chunk: Raw SSE data chunk

        Returns:
            List of SSEParseResult for each complete event in the chunk.
            The accumulated_text in each result is the running total up to that point.
        """
        self.buffer += chunk
        results = []

        # Split by newlines to get SSE events
        lines = self.buffer.split("\n")
        self.buffer = lines[-1]  # Keep incomplete line in buffer

        for line in lines[:-1]:
            line = line.strip()
            result = parse_sse_line(line, self.accumulated_text)

            # Update our accumulated text for next iteration
            self.accumulated_text = result.accumulated_text

            # Only emit results for meaningful events (data lines that changed state)
            if line.startswith("data:"):
                results.append(result)

                # If done or error, stop processing
                if result.is_done or result.error:
                    break

        return results

    def finalize(self) -> SSEParseResult | None:
        """
        Process any remaining data in the buffer.

        Call this after the stream ends to handle any incomplete final event.

        Returns:
            Final SSEParseResult if there was buffered data, None otherwise.
        """
        if not self.buffer.strip():
            return None

        line = self.buffer.strip()
        self.buffer = ""

        if line.startswith("data:"):
            result = parse_sse_line(line, self.accumulated_text)
            self.accumulated_text = result.accumulated_text
            return result

        return None

    def get_text(self) -> str:
        """Get the current accumulated text."""
        return self.accumulated_text.strip()
