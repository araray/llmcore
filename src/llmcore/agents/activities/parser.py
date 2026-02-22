# src/llmcore/agents/activities/parser.py
"""
Activity Request Parser.

Parses activity requests from LLM output using XML format. The parser is designed
to be forgiving of malformed XML while still extracting valid activity requests.

Why XML?
- Easier to parse when malformed (forgiving)
- Distinct from natural language
- Self-closing tags simplify boundary detection
- Works with regex fallbacks
- Clear separation of activities from reasoning

XML Format:
    <activity_request>
        <activity>file_search</activity>
        <parameters>
            <path>/var/log</path>
            <pattern>*.log</pattern>
        </parameters>
        <target>docker:agent-sandbox</target>
        <reason>Find application logs for error analysis</reason>
    </activity_request>

References:
    - Master Plan: Section 9.2 (Activity Request Protocol)
    - Technical Spec: Section 5.4.2 (Request Parsing)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from .schema import ActivityRequest, ExecutionTarget

logger = logging.getLogger(__name__)


# =============================================================================
# PARSE RESULT
# =============================================================================


@dataclass
class ParseResult:
    """
    Result of parsing LLM output for activity requests.

    Contains extracted requests, remaining text, and any errors encountered.
    """

    requests: list[ActivityRequest] = field(default_factory=list)
    remaining_text: str = ""
    errors: list[str] = field(default_factory=list)
    raw_xml_blocks: list[str] = field(default_factory=list)

    @property
    def has_requests(self) -> bool:
        """Check if any requests were parsed."""
        return len(self.requests) > 0

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


# =============================================================================
# ACTIVITY REQUEST PARSER
# =============================================================================


class ActivityRequestParser:
    """
    Parse activity requests from LLM output.

    The parser uses a multi-stage approach:
    1. Extract XML blocks using regex
    2. Parse each block with ElementTree
    3. Fall back to regex parsing if ET fails
    4. Validate extracted data

    Example:
        >>> parser = ActivityRequestParser()
        >>> result = parser.parse('''
        ...     I'll search for log files.
        ...     <activity_request>
        ...         <activity>file_search</activity>
        ...         <parameters>
        ...             <path>/var/log</path>
        ...             <pattern>*.log</pattern>
        ...         </parameters>
        ...     </activity_request>
        ... ''')
        >>> result.requests[0].activity
        'file_search'
    """

    # Patterns for extracting XML blocks
    ACTIVITY_BLOCK_PATTERN = re.compile(
        r"<activity_request\s*>(.*?)</activity_request>",
        re.DOTALL | re.IGNORECASE,
    )

    # Alternative patterns for malformed XML
    ACTIVITY_SIMPLE_PATTERN = re.compile(
        r"<activity\s*>(.*?)</activity>",
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for parameters block
    PARAMETERS_PATTERN = re.compile(
        r"<parameters\s*>(.*?)</parameters>",
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for individual parameter tags
    PARAM_TAG_PATTERN = re.compile(
        r"<(\w+)>(.*?)</\1>",
        re.DOTALL,
    )

    # Pattern for target specification
    TARGET_PATTERN = re.compile(
        r"<target\s*>(.*?)</target>",
        re.DOTALL | re.IGNORECASE,
    )

    # Pattern for reason
    REASON_PATTERN = re.compile(
        r"<reason\s*>(.*?)</reason>",
        re.DOTALL | re.IGNORECASE,
    )

    # Final answer patterns (should not be parsed as activities)
    FINAL_ANSWER_PATTERNS = [
        re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
        re.compile(r"FINAL ANSWER:", re.IGNORECASE),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        allow_multiple: bool = True,
        max_requests: int = 10,
    ):
        """
        Initialize the parser.

        Args:
            strict_mode: If True, reject malformed XML instead of using fallbacks
            allow_multiple: If True, allow multiple activity requests per parse
            max_requests: Maximum number of requests to extract per parse
        """
        self.strict_mode = strict_mode
        self.allow_multiple = allow_multiple
        self.max_requests = max_requests

    def parse(self, text: str) -> ParseResult:
        """
        Parse activity requests from LLM output.

        Args:
            text: LLM output text potentially containing activity requests

        Returns:
            ParseResult with extracted requests and remaining text
        """
        if not text or not text.strip():
            return ParseResult(remaining_text=text)

        result = ParseResult()
        remaining = text

        # Extract all activity_request blocks
        matches = list(self.ACTIVITY_BLOCK_PATTERN.finditer(text))

        if not matches:
            # No activity blocks found - check for final answer
            result.remaining_text = text
            return result

        # Track positions to build remaining text
        positions = []

        for i, match in enumerate(matches):
            if i >= self.max_requests:
                result.errors.append(
                    f"Maximum requests ({self.max_requests}) exceeded, ignoring additional"
                )
                break

            positions.append((match.start(), match.end()))
            xml_content = match.group(0)
            result.raw_xml_blocks.append(xml_content)

            # Try to parse the block
            request, error = self._parse_block(xml_content)

            if request:
                result.requests.append(request)
                if not self.allow_multiple:
                    break
            elif error:
                result.errors.append(error)

        # Build remaining text (text outside activity blocks)
        remaining_parts = []
        last_end = 0
        for start, end in positions:
            if start > last_end:
                remaining_parts.append(text[last_end:start].strip())
            last_end = end
        if last_end < len(text):
            remaining_parts.append(text[last_end:].strip())

        result.remaining_text = "\n".join(p for p in remaining_parts if p)

        return result

    def _parse_block(self, xml_content: str) -> tuple[ActivityRequest | None, str | None]:
        """
        Parse a single activity_request block.

        Args:
            xml_content: XML content of the activity_request block

        Returns:
            Tuple of (ActivityRequest, None) on success or (None, error_message) on failure
        """
        # Try ElementTree first
        try:
            return self._parse_with_et(xml_content), None
        except ET.ParseError as e:
            if self.strict_mode:
                return None, f"XML parse error: {e}"
            logger.debug(f"ET parse failed, trying regex fallback: {e}")

        # Fall back to regex parsing
        try:
            return self._parse_with_regex(xml_content), None
        except Exception as e:
            return None, f"Failed to parse activity request: {e}"

    def _parse_with_et(self, xml_content: str) -> ActivityRequest:
        """
        Parse activity request using ElementTree.

        Args:
            xml_content: Well-formed XML content

        Returns:
            Parsed ActivityRequest

        Raises:
            ET.ParseError: If XML is malformed
            ValueError: If required fields are missing
        """
        root = ET.fromstring(xml_content)

        # Extract activity name
        activity_elem = root.find("activity")
        if activity_elem is None:
            raise ValueError("Missing <activity> element")
        activity_name = (activity_elem.text or "").strip()

        if not activity_name:
            raise ValueError("Empty activity name")

        # Extract parameters
        parameters: dict[str, Any] = {}
        params_elem = root.find("parameters")
        if params_elem is not None:
            for param in params_elem:
                param_name = param.tag
                param_value = self._parse_param_value(param)
                parameters[param_name] = param_value

        # Extract target
        target = ExecutionTarget.DOCKER  # Default
        target_elem = root.find("target")
        if target_elem is not None and target_elem.text:
            target_str = target_elem.text.strip().lower()
            target = self._parse_target(target_str)

        # Extract reason
        reason: str | None = None
        reason_elem = root.find("reason")
        if reason_elem is not None and reason_elem.text:
            reason = reason_elem.text.strip()

        return ActivityRequest(
            activity=activity_name,
            parameters=parameters,
            target=target,
            reason=reason,
        )

    def _parse_param_value(self, element: ET.Element) -> Any:
        """
        Parse parameter value, handling nested structures.

        Args:
            element: XML element containing the value

        Returns:
            Parsed value (str, list, or dict)
        """
        # Check if element has children (nested structure)
        if len(element) > 0:
            # Nested elements - could be list or dict
            # Check if all children have same tag (likely a list)
            tags = [child.tag for child in element]
            if len(set(tags)) == 1 and len(tags) > 1:
                # Same tag repeated - treat as list
                return [self._parse_param_value(child) for child in element]
            else:
                # Different tags - treat as dict
                return {child.tag: self._parse_param_value(child) for child in element}
        else:
            # Leaf element
            text = (element.text or "").strip()

            # Try to parse as JSON-like types
            if text.lower() in ("true", "false"):
                return text.lower() == "true"
            try:
                return int(text)
            except ValueError:
                pass
            try:
                return float(text)
            except ValueError:
                pass

            return text

    def _parse_with_regex(self, xml_content: str) -> ActivityRequest:
        """
        Parse activity request using regex fallback.

        Args:
            xml_content: Potentially malformed XML content

        Returns:
            Parsed ActivityRequest

        Raises:
            ValueError: If required fields cannot be extracted
        """
        # Extract activity name
        activity_match = self.ACTIVITY_SIMPLE_PATTERN.search(xml_content)
        if not activity_match:
            raise ValueError("Could not find <activity> tag")

        activity_name = activity_match.group(1).strip()
        if not activity_name:
            raise ValueError("Empty activity name")

        # Extract parameters
        parameters: dict[str, Any] = {}
        params_match = self.PARAMETERS_PATTERN.search(xml_content)
        if params_match:
            params_content = params_match.group(1)
            for param_match in self.PARAM_TAG_PATTERN.finditer(params_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                # Skip if it's a nested element
                if "<" in param_value:
                    continue

                # Try type conversion
                if param_value.lower() in ("true", "false"):
                    parameters[param_name] = param_value.lower() == "true"
                else:
                    try:
                        parameters[param_name] = int(param_value)
                    except ValueError:
                        try:
                            parameters[param_name] = float(param_value)
                        except ValueError:
                            parameters[param_name] = param_value

        # Extract target
        target = ExecutionTarget.DOCKER
        target_match = self.TARGET_PATTERN.search(xml_content)
        if target_match:
            target = self._parse_target(target_match.group(1).strip())

        # Extract reason
        reason: str | None = None
        reason_match = self.REASON_PATTERN.search(xml_content)
        if reason_match:
            reason = reason_match.group(1).strip()

        return ActivityRequest(
            activity=activity_name,
            parameters=parameters,
            target=target,
            reason=reason,
        )

    def _parse_target(self, target_str: str) -> ExecutionTarget:
        """
        Parse execution target from string.

        Args:
            target_str: Target string (e.g., "docker:sandbox", "local", "vm")

        Returns:
            ExecutionTarget enum value
        """
        target_lower = target_str.lower()

        # Handle prefixed targets (e.g., "docker:sandbox")
        if ":" in target_lower:
            target_lower = target_lower.split(":")[0]

        target_map = {
            "local": ExecutionTarget.LOCAL,
            "docker": ExecutionTarget.DOCKER,
            "container": ExecutionTarget.DOCKER,
            "vm": ExecutionTarget.VM,
            "sandbox": ExecutionTarget.DOCKER,
            "remote": ExecutionTarget.REMOTE,
            "dry_run": ExecutionTarget.DRY_RUN,
            "dry-run": ExecutionTarget.DRY_RUN,
            "dryrun": ExecutionTarget.DRY_RUN,
        }

        return target_map.get(target_lower, ExecutionTarget.DOCKER)

    def is_final_answer(self, text: str) -> bool:
        """
        Check if text contains a final answer marker.

        Detects final answers in multiple formats:
        - <final_answer>...</final_answer>
        - <answer>...</answer>
        - FINAL ANSWER: ...
        - <activity>final_answer</activity> (activity system)
        - <activity>finish</activity> (activity system)

        Args:
            text: Text to check

        Returns:
            True if text appears to be a final answer
        """
        # Check standard final answer patterns
        for pattern in self.FINAL_ANSWER_PATTERNS:
            if pattern.search(text):
                return True

        # Check for activity-based final answer (activity name = final_answer or finish)
        activity_match = self.ACTIVITY_SIMPLE_PATTERN.search(text)
        if activity_match:
            activity_name = activity_match.group(1).strip().lower()
            if activity_name in ("final_answer", "finish", "complete", "done"):
                return True

        return False

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extract final answer content if present.

        Handles multiple formats:
        - <final_answer>content</final_answer>
        - <answer>content</answer>
        - <activity>final_answer</activity> with <parameters>

        Args:
            text: Text potentially containing a final answer

        Returns:
            Final answer content, or None if not found
        """
        # Check standard final answer patterns first
        for pattern in self.FINAL_ANSWER_PATTERNS[:2]:  # Only structured patterns
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        # Check for activity-based final answer
        activity_match = self.ACTIVITY_SIMPLE_PATTERN.search(text)
        if activity_match:
            activity_name = activity_match.group(1).strip().lower()
            if activity_name in ("final_answer", "finish", "complete", "done"):
                # Try to extract content from the activity request
                parse_result = self.parse(text)
                if parse_result.has_requests:
                    request = parse_result.requests[0]
                    # Look for answer content in parameters
                    params = request.parameters
                    if params:
                        # Try common parameter names for the answer
                        for key in ("result", "answer", "content", "response", "output", "r"):
                            if key in params:
                                return str(params[key])
                        # If no known key, return first parameter value
                        first_value = next(iter(params.values()), None)
                        if first_value:
                            return str(first_value)

        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def parse_activity_requests(text: str, **kwargs) -> ParseResult:
    """
    Parse activity requests from text using default parser.

    Args:
        text: LLM output text
        **kwargs: Additional parser configuration

    Returns:
        ParseResult with extracted requests
    """
    parser = ActivityRequestParser(**kwargs)
    return parser.parse(text)


def has_activity_request(text: str) -> bool:
    """
    Quick check if text contains any activity requests.

    Args:
        text: Text to check

    Returns:
        True if text contains activity request markers
    """
    return bool(ActivityRequestParser.ACTIVITY_BLOCK_PATTERN.search(text))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ActivityRequestParser",
    "ParseResult",
    "has_activity_request",
    "parse_activity_requests",
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_inputs = [
        # Standard format
        """
        I'll search for log files.
        <activity_request>
            <activity>file_search</activity>
            <parameters>
                <path>/var/log</path>
                <pattern>*.log</pattern>
            </parameters>
            <reason>Find application logs for error analysis</reason>
        </activity_request>
        """,
        # Multiple requests
        """
        Let me read and then write a file.
        <activity_request>
            <activity>file_read</activity>
            <parameters>
                <path>/etc/config.yaml</path>
            </parameters>
        </activity_request>
        <activity_request>
            <activity>file_write</activity>
            <parameters>
                <path>/tmp/output.txt</path>
                <content>Hello World</content>
            </parameters>
        </activity_request>
        """,
        # Malformed but parseable
        """
        <activity_request>
        <activity>python_exec</activity>
        <parameters>
        <code>print("hello")</code>
        </parameters>
        </activity_request>
        """,
        # No activities
        "Just a regular response with no activities.",
        # Final answer
        "<final_answer>The result is 42.</final_answer>",
    ]

    parser = ActivityRequestParser()

    for i, test_input in enumerate(test_inputs):
        print(f"\n{'=' * 60}")
        print(f"Test {i + 1}")
        print("=" * 60)

        result = parser.parse(test_input)

        print(f"Requests found: {len(result.requests)}")
        for j, req in enumerate(result.requests):
            print(f"  {j + 1}. {req.activity}: {req.parameters}")

        if result.errors:
            print(f"Errors: {result.errors}")

        if result.remaining_text:
            print(f"Remaining: {result.remaining_text[:50]}...")

        if parser.is_final_answer(test_input):
            answer = parser.extract_final_answer(test_input)
            print(f"Final answer: {answer}")

    print("\n✅ Parser self-test complete!")
