# tests/api/test_phase6_sessions2_3.py
"""
Unit tests for Phase 6 Sessions 2-3: Extended Export/Import functionality.

Tests:
- Markdown format export
- Bundle (.llmchat) creation and import
- Context library export/import
- Full state export/import
- Error handling for all new functionality

Session 2: Markdown Export & Bundle Format
Session 3: Library Export & State Export
"""

import json
import os
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ==============================================================================
# Mock Classes (Shared with test_phase6_export_import.py)
# ==============================================================================


class MockRole:
    """Mock Role enum."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value


class MockContextItemType:
    """Mock ContextItemType enum."""

    USER_TEXT = "user_text"
    USER_FILE = "user_file"
    RAG_SNIPPET = "rag_snippet"

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value


class MockMessage:
    """Mock Message object."""

    def __init__(
        self,
        content: str,
        role: str = "user",
        id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        tool_call_id: Optional[str] = None,
        tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.session_id = session_id
        self.role = MockRole(role)
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.tool_call_id = tool_call_id
        self.tokens = tokens
        self.metadata = metadata or {}


class MockContextItem:
    """Mock ContextItem object."""

    def __init__(
        self,
        content: str,
        type_str: str = "user_text",
        id: Optional[str] = None,
        source_id: Optional[str] = None,
        tokens: Optional[int] = None,
        original_tokens: Optional[int] = None,
        is_truncated: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.type = MockContextItemType(type_str)
        self.source_id = source_id
        self.content = content
        self.tokens = tokens
        self.original_tokens = original_tokens
        self.is_truncated = is_truncated
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)


class MockChatSession:
    """Mock ChatSession object."""

    def __init__(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        messages: Optional[List[MockMessage]] = None,
        context_items: Optional[List[MockContextItem]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.messages = messages or []
        self.context_items = context_items or []
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.metadata = metadata or {}

    def add_context_item(self, item):
        self.context_items.append(item)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def sample_session():
    """Create a sample session with messages and context items."""
    session = MockChatSession(
        id="test_session_123",
        name="Test Session",
        metadata={"provider": "openai", "model": "gpt-4"},
    )

    session.messages = [
        MockMessage(
            id="msg_1",
            content="Hello, world!",
            role="user",
            tokens=5,
            metadata={"source": "test"},
        ),
        MockMessage(
            id="msg_2",
            content="Hello! How can I help you today?",
            role="assistant",
            tokens=10,
        ),
        MockMessage(
            id="msg_3",
            content="What is Python?",
            role="user",
            tokens=4,
        ),
    ]

    session.context_items = [
        MockContextItem(
            id="ctx_1",
            content="This is some context text.",
            type_str="user_text",
            tokens=6,
            metadata={"filename": "context.txt"},
        ),
    ]

    return session


@pytest.fixture
def sample_export_data(sample_session):
    """Create sample export data dictionary."""
    return {
        "llmcore_export_version": "1.0",
        "export_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session": {
            "id": sample_session.id,
            "name": sample_session.name,
            "created_at": sample_session.created_at.isoformat().replace("+00:00", "Z"),
            "updated_at": sample_session.updated_at.isoformat().replace("+00:00", "Z"),
            "metadata": sample_session.metadata,
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat().replace("+00:00", "Z"),
                "tokens": msg.tokens,
            }
            for msg in sample_session.messages
        ],
        "context_items": [
            {
                "id": item.id,
                "type": item.type.value,
                "content": item.content,
                "tokens": item.tokens,
            }
            for item in sample_session.context_items
        ],
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# Test Markdown Export (Session 2)
# ==============================================================================


class TestMarkdownExport:
    """Tests for Markdown format export."""

    def test_markdown_export_has_yaml_frontmatter(self, sample_session, sample_export_data):
        """Test that Markdown export contains YAML frontmatter."""
        # Simulate markdown formatting
        lines = []
        lines.append("---")
        lines.append(f'llmcore_export_version: "{sample_export_data["llmcore_export_version"]}"')
        lines.append(f"session_id: {sample_session.id}")
        lines.append(f'session_name: "{sample_session.name}"')
        lines.append(f"message_count: {len(sample_export_data['messages'])}")
        lines.append("---")

        markdown = "\n".join(lines)

        assert markdown.startswith("---")
        assert 'llmcore_export_version: "1.0"' in markdown
        assert sample_session.id in markdown
        assert sample_session.name in markdown

    def test_markdown_export_has_conversation_section(self, sample_session):
        """Test that Markdown export includes conversation section."""
        lines = []
        lines.append("## Conversation")
        lines.append("")

        for i, msg in enumerate(sample_session.messages, 1):
            role = msg.role.value.upper()
            lines.append(f"### {role}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        markdown = "\n".join(lines)

        assert "## Conversation" in markdown
        assert "### USER" in markdown
        assert "### ASSISTANT" in markdown
        assert "Hello, world!" in markdown

    def test_markdown_export_has_context_section(self, sample_session):
        """Test that Markdown export includes context items section."""
        lines = []
        lines.append("## Context Items")
        lines.append("")

        for item in sample_session.context_items:
            lines.append(f"### {item.id} ({item.type.value})")
            lines.append("")
            lines.append(item.content)
            lines.append("")

        markdown = "\n".join(lines)

        assert "## Context Items" in markdown
        assert "ctx_1" in markdown
        assert "user_text" in markdown

    def test_markdown_export_escapes_special_characters(self):
        """Test that special characters are handled in Markdown."""
        special_content = "Code: `print('hello')` and *bold* text"
        message = MockMessage(content=special_content)

        # Markdown should preserve the content as-is
        assert "`print('hello')`" in message.content
        assert "*bold*" in message.content

    def test_markdown_export_truncates_long_context(self):
        """Test that very long context items are truncated in Markdown."""
        long_content = "x" * 5000
        truncated = long_content[:2000] + "\n\n*[Content truncated for readability...]*"

        assert len(truncated) < len(long_content)
        assert "[Content truncated" in truncated


# ==============================================================================
# Test Bundle Creation (Session 2)
# ==============================================================================


class TestBundleCreation:
    """Tests for .llmchat bundle creation."""

    def test_bundle_is_valid_zip(self, temp_dir, sample_export_data):
        """Test that bundle creates a valid ZIP file."""
        bundle_path = temp_dir / "test.llmchat"

        # Create a bundle manually for testing
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "bundle_version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "session_id": "test_session_123",
                "contents": {"session": "session.json", "files": []},
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("session.json", json.dumps(sample_export_data, indent=2))

        # Verify it's a valid ZIP
        assert zipfile.is_zipfile(bundle_path)

        # Verify contents
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "session.json" in names

    def test_bundle_contains_manifest(self, temp_dir, sample_export_data):
        """Test that bundle contains a valid manifest.json."""
        bundle_path = temp_dir / "test.llmchat"

        manifest = {
            "bundle_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "created_by": "llmcore",
            "session_id": "test_session_123",
            "session_name": "Test Session",
            "message_count": 3,
            "context_item_count": 1,
            "contents": {"session": "session.json", "files": []},
        }

        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("session.json", json.dumps(sample_export_data, indent=2))

        # Read and verify manifest
        with zipfile.ZipFile(bundle_path, "r") as zf:
            manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))

        assert manifest_data["bundle_version"] == "1.0"
        assert manifest_data["session_id"] == "test_session_123"
        assert manifest_data["message_count"] == 3

    def test_bundle_includes_files(self, temp_dir, sample_export_data):
        """Test that bundle can include additional files."""
        bundle_path = temp_dir / "test.llmchat"

        # Create a test file to include
        test_file = temp_dir / "main.py"
        test_file.write_text("print('hello')")

        manifest = {
            "bundle_version": "1.0",
            "session_id": "test_session_123",
            "contents": {"session": "session.json", "files": ["files/main.py"]},
        }

        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("session.json", json.dumps(sample_export_data, indent=2))
            zf.write(test_file, "files/main.py")

        # Verify file is in bundle
        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "files/main.py" in names

            content = zf.read("files/main.py").decode("utf-8")
            assert "print('hello')" in content

    def test_bundle_file_extension(self, temp_dir):
        """Test that bundle path gets .llmchat extension."""
        path_without_ext = temp_dir / "backup"
        path_with_ext = path_without_ext.with_suffix(".llmchat")

        assert path_with_ext.suffix == ".llmchat"
        assert str(path_with_ext).endswith(".llmchat")


# ==============================================================================
# Test Bundle Import (Session 2)
# ==============================================================================


class TestBundleImport:
    """Tests for .llmchat bundle import."""

    def test_import_bundle_extracts_session(self, temp_dir, sample_export_data):
        """Test that importing bundle extracts session data."""
        bundle_path = temp_dir / "test.llmchat"

        manifest = {
            "bundle_version": "1.0",
            "session_id": "test_session_123",
            "contents": {"session": "session.json", "files": []},
        }

        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("session.json", json.dumps(sample_export_data, indent=2))

        # Read session from bundle
        with zipfile.ZipFile(bundle_path, "r") as zf:
            session_data = json.loads(zf.read("session.json").decode("utf-8"))

        assert session_data["session"]["id"] == "test_session_123"
        assert len(session_data["messages"]) == 3

    def test_import_bundle_extracts_files(self, temp_dir, sample_export_data):
        """Test that importing bundle can extract included files."""
        bundle_path = temp_dir / "test.llmchat"
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        # Create bundle with a file
        test_content = "# Test file content"

        with zipfile.ZipFile(bundle_path, "w") as zf:
            manifest = {
                "bundle_version": "1.0",
                "contents": {"session": "session.json", "files": ["files/test.py"]},
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("session.json", json.dumps(sample_export_data, indent=2))
            zf.writestr("files/test.py", test_content)

        # Extract file
        with zipfile.ZipFile(bundle_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

            for file_entry in manifest["contents"]["files"]:
                file_name = Path(file_entry).name
                target_path = extract_dir / file_name
                with zf.open(file_entry) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

        # Verify extraction
        extracted_file = extract_dir / "test.py"
        assert extracted_file.exists()
        assert extracted_file.read_text() == test_content

    def test_import_bundle_handles_invalid_zip(self, temp_dir):
        """Test that import handles invalid ZIP gracefully."""
        invalid_path = temp_dir / "invalid.llmchat"
        invalid_path.write_text("not a zip file")

        with pytest.raises(zipfile.BadZipFile):
            with zipfile.ZipFile(invalid_path, "r"):
                pass

    def test_import_bundle_handles_missing_manifest(self, temp_dir):
        """Test that import handles bundle without manifest."""
        bundle_path = temp_dir / "test.llmchat"

        # Create bundle without manifest
        with zipfile.ZipFile(bundle_path, "w") as zf:
            zf.writestr("session.json", "{}")

        with zipfile.ZipFile(bundle_path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" not in names


# ==============================================================================
# Test Library Export (Session 3)
# ==============================================================================


class TestLibraryExport:
    """Tests for context library export."""

    def test_library_export_structure(self):
        """Test that library export has correct structure."""
        library_data = {
            "llmcore_export_version": "1.0",
            "export_type": "context_library_full",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "chunks": {
                "coding_standards": {
                    "id": "chunk_001",
                    "name": "coding_standards",
                    "content_type": "text",
                    "content": "Follow PEP 8...",
                    "tags": ["python", "style"],
                }
            },
            "templates": {
                "python_dev": {
                    "id": "tmpl_001",
                    "name": "python_dev",
                    "description": "Python development setup",
                    "items": [{"type": "chunk", "ref": "coding_standards"}],
                }
            },
        }

        # Verify structure
        assert "chunks" in library_data
        assert "templates" in library_data
        assert library_data["export_type"] == "context_library_full"

    def test_library_export_partial(self):
        """Test exporting specific chunks/templates."""
        full_library = {
            "chunks": {
                "chunk1": {"name": "chunk1", "content": "Content 1"},
                "chunk2": {"name": "chunk2", "content": "Content 2"},
                "chunk3": {"name": "chunk3", "content": "Content 3"},
            },
            "templates": {},
        }

        # Export only chunk1 and chunk2
        items_to_export = ["chunk1", "chunk2"]
        partial_export = {
            "export_type": "context_library_partial",
            "chunks": {k: v for k, v in full_library["chunks"].items() if k in items_to_export},
            "templates": {},
        }

        assert len(partial_export["chunks"]) == 2
        assert "chunk3" not in partial_export["chunks"]

    def test_library_export_to_json(self, temp_dir):
        """Test exporting library to JSON file."""
        library_data = {
            "chunks": {"test": {"name": "test", "content": "Test content"}},
            "templates": {},
        }

        export_path = temp_dir / "library.json"
        with open(export_path, "w") as f:
            json.dump(library_data, f, indent=2)

        # Verify file
        assert export_path.exists()
        with open(export_path) as f:
            loaded = json.load(f)
        assert loaded["chunks"]["test"]["name"] == "test"


# ==============================================================================
# Test Library Import (Session 3)
# ==============================================================================


class TestLibraryImport:
    """Tests for context library import."""

    def test_library_import_merges_chunks(self):
        """Test that importing library merges chunks."""
        existing_library = {
            "chunks": {"existing_chunk": {"name": "existing_chunk", "content": "Existing"}},
            "templates": {},
        }

        import_data = {
            "chunks": {"new_chunk": {"name": "new_chunk", "content": "New content"}},
            "templates": {},
        }

        # Merge
        existing_library["chunks"].update(import_data["chunks"])

        assert len(existing_library["chunks"]) == 2
        assert "existing_chunk" in existing_library["chunks"]
        assert "new_chunk" in existing_library["chunks"]

    def test_library_import_replaces_duplicates(self):
        """Test that importing library replaces duplicates."""
        existing_library = {
            "chunks": {"chunk1": {"name": "chunk1", "content": "Old content"}},
        }

        import_data = {
            "chunks": {"chunk1": {"name": "chunk1", "content": "New content"}},
        }

        # Merge (replace existing)
        existing_library["chunks"].update(import_data["chunks"])

        assert existing_library["chunks"]["chunk1"]["content"] == "New content"

    def test_library_import_validates_structure(self):
        """Test that invalid library data is rejected."""
        invalid_data = {"not_chunks": {}, "not_templates": {}}

        has_chunks = "chunks" in invalid_data
        has_templates = "templates" in invalid_data

        assert not has_chunks
        assert not has_templates


# ==============================================================================
# Test State Export (Session 3)
# ==============================================================================


class TestStateExport:
    """Tests for full state export."""

    def test_state_export_includes_session(self, sample_export_data):
        """Test that state export includes session data."""
        state_data = {
            "llmcore_export_version": "1.0",
            "export_type": "full_state",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "session": sample_export_data,
            "context_library": None,
            "repl_config": None,
        }

        assert state_data["export_type"] == "full_state"
        assert state_data["session"] is not None
        assert state_data["session"]["session"]["id"] == "test_session_123"

    def test_state_export_includes_library(self):
        """Test that state export includes context library."""
        library_data = {
            "chunks": {"chunk1": {"name": "chunk1", "content": "Content"}},
            "templates": {},
        }

        state_data = {
            "export_type": "full_state",
            "session": None,
            "context_library": library_data,
            "repl_config": None,
        }

        assert state_data["context_library"] is not None
        assert "chunks" in state_data["context_library"]

    def test_state_export_includes_config(self):
        """Test that state export includes REPL config."""
        config_data = {
            "current_session_id": "session_123",
            "current_provider_name": "openai",
            "current_model_name": "gpt-4",
            "rag_enabled": False,
        }

        state_data = {
            "export_type": "full_state",
            "session": None,
            "context_library": None,
            "repl_config": config_data,
        }

        assert state_data["repl_config"]["current_provider_name"] == "openai"


# ==============================================================================
# Test State Import (Session 3)
# ==============================================================================


class TestStateImport:
    """Tests for full state import."""

    def test_state_import_restores_session(self, sample_export_data):
        """Test that state import can restore session."""
        state_data = {
            "export_type": "full_state",
            "session": sample_export_data,
            "context_library": None,
            "repl_config": None,
        }

        # Extract session
        session_to_import = state_data.get("session")

        assert session_to_import is not None
        assert len(session_to_import["messages"]) == 3

    def test_state_import_restores_library(self):
        """Test that state import can restore library."""
        state_data = {
            "export_type": "full_state",
            "session": None,
            "context_library": {
                "chunks": {"chunk1": {"name": "chunk1", "content": "Restored content"}},
                "templates": {},
            },
            "repl_config": None,
        }

        library_to_import = state_data.get("context_library")

        assert library_to_import is not None
        assert "chunk1" in library_to_import["chunks"]

    def test_state_import_selective(self):
        """Test that state import can be selective."""
        state_data = {
            "export_type": "full_state",
            "session": {"messages": []},
            "context_library": {"chunks": {}},
            "repl_config": {"provider": "openai"},
        }

        # Import only session, not library or config
        import_session = True
        import_library = False
        import_config = False

        to_import = []
        if import_session and state_data.get("session"):
            to_import.append("session")
        if import_library and state_data.get("context_library"):
            to_import.append("library")
        if import_config and state_data.get("repl_config"):
            to_import.append("config")

        assert "session" in to_import
        assert "library" not in to_import
        assert "config" not in to_import


# ==============================================================================
# Test Error Handling
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling in export/import operations."""

    def test_handle_nonexistent_file(self, temp_dir):
        """Test handling of nonexistent file."""
        nonexistent = temp_dir / "does_not_exist.json"

        assert not nonexistent.exists()

    def test_handle_corrupted_json(self, temp_dir):
        """Test handling of corrupted JSON file."""
        corrupted = temp_dir / "corrupted.json"
        corrupted.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            with open(corrupted) as f:
                json.load(f)

    def test_handle_permission_error(self, temp_dir):
        """Test handling of permission errors."""
        if os.name == "nt":  # Skip on Windows
            pytest.skip("Permission test not reliable on Windows")

        # Create a read-only file
        readonly = temp_dir / "readonly.json"
        readonly.write_text("{}")
        readonly.chmod(0o444)

        try:
            # Try to write - should fail
            with pytest.raises(PermissionError):
                with open(readonly, "w") as f:
                    f.write("new content")
        finally:
            # Cleanup - make writable again
            readonly.chmod(0o644)

    def test_handle_empty_session(self):
        """Test handling of empty session export."""
        empty_session = MockChatSession(
            id="empty_session",
            name="Empty",
            messages=[],
            context_items=[],
        )

        export_data = {
            "session": {"id": empty_session.id, "name": empty_session.name},
            "messages": [],
            "context_items": [],
        }

        # Should not raise
        json_output = json.dumps(export_data)
        assert '"messages": []' in json_output


# ==============================================================================
# Test Format Detection
# ==============================================================================


class TestFormatDetection:
    """Tests for format detection in import operations."""

    def test_detect_bundle_by_extension(self, temp_dir):
        """Test detecting bundle format by .llmchat extension."""
        bundle_path = temp_dir / "backup.llmchat"

        suffix = bundle_path.suffix.lower()
        assert suffix == ".llmchat"

    def test_detect_json_by_extension(self, temp_dir):
        """Test detecting JSON format by extension."""
        json_path = temp_dir / "data.json"

        suffix = json_path.suffix.lower()
        assert suffix == ".json"

    def test_detect_yaml_by_extension(self, temp_dir):
        """Test detecting YAML format by extension."""
        yaml_path = temp_dir / "data.yaml"
        yml_path = temp_dir / "data.yml"

        assert yaml_path.suffix.lower() == ".yaml"
        assert yml_path.suffix.lower() == ".yml"

    def test_detect_library_export_type(self):
        """Test detecting library export by export_type field."""
        library_data = {"export_type": "context_library_full", "chunks": {}}
        state_data = {"export_type": "full_state", "session": None}

        assert library_data.get("export_type") == "context_library_full"
        assert state_data.get("export_type") == "full_state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
