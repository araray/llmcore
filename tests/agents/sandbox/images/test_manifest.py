# tests/agents/sandbox/images/test_manifest.py
"""
Tests for the manifest parsing module.

Tests cover:
    - Loading manifests from files
    - Loading manifests from strings
    - Parsing and validation
    - Builtin manifest access
    - Error handling
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from llmcore.agents.sandbox.images import (
    MANIFEST_PATH_IN_CONTAINER,
    AccessMode,
    ImageCapability,
    ImageManifest,
    ImageTier,
    ManifestNotFoundError,
    ManifestParseError,
    ManifestValidationError,
    get_builtin_manifest,
    list_builtin_manifests,
    load_manifest_from_file,
    load_manifest_from_string,
    parse_manifest,
    save_manifest_to_file,
    validate_manifest,
)

# ==============================================================================
# load_manifest_from_file Tests
# ==============================================================================


class TestLoadManifestFromFile:
    """Tests for load_manifest_from_file function."""

    def test_load_valid_manifest(self, manifest_file: Path):
        """Test loading a valid manifest file."""
        manifest = load_manifest_from_file(manifest_file)
        assert manifest.name == "llmcore-sandbox-test"
        assert manifest.version == "1.0.0"
        assert manifest.tier == ImageTier.SPECIALIZED

    def test_load_nonexistent_file(self, temp_dir: Path):
        """Test loading from nonexistent file raises error."""
        nonexistent = temp_dir / "nonexistent.json"
        with pytest.raises(ManifestNotFoundError) as exc_info:
            load_manifest_from_file(nonexistent)
        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_json(self, invalid_manifest_file: Path):
        """Test loading invalid JSON raises error."""
        with pytest.raises(ManifestParseError) as exc_info:
            load_manifest_from_file(invalid_manifest_file)
        assert "json" in str(exc_info.value).lower()

    def test_load_accepts_string_path(self, manifest_file: Path):
        """Test that string paths are accepted."""
        manifest = load_manifest_from_file(str(manifest_file))
        assert manifest.name == "llmcore-sandbox-test"


# ==============================================================================
# load_manifest_from_string Tests
# ==============================================================================


class TestLoadManifestFromString:
    """Tests for load_manifest_from_string function."""

    def test_load_valid_json(self, sample_manifest_dict: Dict[str, Any]):
        """Test loading from valid JSON string."""
        json_str = json.dumps(sample_manifest_dict)
        manifest = load_manifest_from_string(json_str)
        assert manifest.name == "llmcore-sandbox-test"
        assert manifest.version == "1.0.0"

    def test_load_minimal_manifest(self):
        """Test loading minimal valid manifest."""
        json_str = '{"name": "test", "version": "1.0.0"}'
        manifest = load_manifest_from_string(json_str)
        assert manifest.name == "test"
        assert manifest.version == "1.0.0"

    def test_load_invalid_json_string(self):
        """Test loading invalid JSON raises error."""
        with pytest.raises(ManifestParseError):
            load_manifest_from_string("not valid json")

    def test_load_empty_string(self):
        """Test loading empty string raises error."""
        with pytest.raises(ManifestParseError):
            load_manifest_from_string("")


# ==============================================================================
# parse_manifest Tests
# ==============================================================================


class TestParseManifest:
    """Tests for parse_manifest function."""

    def test_parse_full_manifest(self, sample_manifest_dict: Dict[str, Any]):
        """Test parsing a complete manifest."""
        manifest = parse_manifest(sample_manifest_dict)
        assert manifest.name == "llmcore-sandbox-test"
        assert manifest.version == "1.0.0"
        assert manifest.tier == ImageTier.SPECIALIZED
        assert ImageCapability.PYTHON in manifest.capabilities

    def test_parse_minimal_manifest(self):
        """Test parsing minimal required fields."""
        data = {"name": "minimal", "version": "0.1.0"}
        manifest = parse_manifest(data)
        assert manifest.name == "minimal"
        assert manifest.version == "0.1.0"
        assert manifest.tier == ImageTier.BASE  # default

    def test_parse_missing_name(self):
        """Test that missing name raises error."""
        data = {"version": "1.0.0"}
        with pytest.raises(ManifestValidationError) as exc_info:
            parse_manifest(data)
        assert "name" in str(exc_info.value).lower()

    def test_parse_missing_version(self):
        """Test that missing version raises error."""
        data = {"name": "test"}
        with pytest.raises(ManifestValidationError) as exc_info:
            parse_manifest(data)
        assert "version" in str(exc_info.value).lower()

    def test_parse_capabilities(self):
        """Test parsing capabilities list."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "capabilities": ["python", "shell", "git"],
        }
        manifest = parse_manifest(data)
        assert ImageCapability.PYTHON in manifest.capabilities
        assert ImageCapability.SHELL in manifest.capabilities
        assert ImageCapability.GIT in manifest.capabilities

    def test_parse_unknown_capabilities_ignored(self):
        """Test that unknown capabilities are silently ignored."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "capabilities": ["python", "unknown_cap", "shell"],
        }
        manifest = parse_manifest(data)
        assert len(manifest.capabilities) == 2
        assert ImageCapability.PYTHON in manifest.capabilities
        assert ImageCapability.SHELL in manifest.capabilities

    def test_parse_resource_limits(self):
        """Test parsing resource limits."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "resource_limits": {
                "memory_limit": "2g",
                "cpu_limit": 4.0,
            },
        }
        manifest = parse_manifest(data)
        assert manifest.resource_limits.memory_limit == "2g"
        assert manifest.resource_limits.cpu_limit == 4.0

    def test_parse_access_mode(self):
        """Test parsing access mode."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "default_access_mode": "full",
        }
        manifest = parse_manifest(data)
        assert manifest.default_access_mode == AccessMode.FULL


# ==============================================================================
# validate_manifest Tests
# ==============================================================================


class TestValidateManifest:
    """Tests for validate_manifest function."""

    def test_validate_valid_manifest(self, sample_manifest: ImageManifest):
        """Test that valid manifest passes validation."""
        # Should not raise
        validate_manifest(sample_manifest)

    def test_validate_empty_name(self):
        """Test that empty name fails validation."""
        manifest = ImageManifest(name="", version="1.0.0", tier=ImageTier.BASE)
        with pytest.raises(ManifestValidationError) as exc_info:
            validate_manifest(manifest)
        assert "name" in str(exc_info.value).lower()

    def test_validate_empty_version(self):
        """Test that empty version fails validation."""
        manifest = ImageManifest(name="test", version="", tier=ImageTier.BASE)
        with pytest.raises(ManifestValidationError) as exc_info:
            validate_manifest(manifest)
        assert "version" in str(exc_info.value).lower()

    def test_validate_invalid_version_format(self):
        """Test that non-semver version fails validation."""
        manifest = ImageManifest(name="test", version="v1", tier=ImageTier.BASE)
        with pytest.raises(ManifestValidationError) as exc_info:
            validate_manifest(manifest)
        assert "version" in str(exc_info.value).lower()

    def test_validate_base_with_llmcore_parent(self):
        """Test that base image with llmcore parent fails."""
        manifest = ImageManifest(
            name="llmcore-sandbox-base2",
            version="1.0.0",
            tier=ImageTier.BASE,
            base_image="llmcore-sandbox-base:1.0.0",
        )
        with pytest.raises(ManifestValidationError):
            validate_manifest(manifest)


# ==============================================================================
# save_manifest_to_file Tests
# ==============================================================================


class TestSaveManifestToFile:
    """Tests for save_manifest_to_file function."""

    def test_save_manifest(self, temp_dir: Path, sample_manifest: ImageManifest):
        """Test saving manifest to file."""
        output_path = temp_dir / "output.json"
        save_manifest_to_file(sample_manifest, output_path)

        assert output_path.exists()

        # Load and verify
        with open(output_path) as f:
            data = json.load(f)
        assert data["name"] == sample_manifest.name
        assert data["version"] == sample_manifest.version

    def test_save_creates_directories(self, temp_dir: Path, sample_manifest: ImageManifest):
        """Test that save creates parent directories."""
        output_path = temp_dir / "nested" / "dir" / "manifest.json"
        save_manifest_to_file(sample_manifest, output_path)
        assert output_path.exists()

    def test_save_overwrite(self, manifest_file: Path, sample_manifest: ImageManifest):
        """Test that save overwrites existing file."""
        # Modify and save
        sample_manifest.description = "Updated description"
        save_manifest_to_file(sample_manifest, manifest_file)

        # Verify
        manifest = load_manifest_from_file(manifest_file)
        assert manifest.description == "Updated description"


# ==============================================================================
# Builtin Manifest Access Tests
# ==============================================================================


class TestBuiltinManifestAccess:
    """Tests for builtin manifest access functions."""

    def test_get_builtin_manifest_by_name(self):
        """Test getting builtin manifest by exact name."""
        manifest = get_builtin_manifest("llmcore-sandbox-python")
        assert manifest is not None
        assert manifest.name == "llmcore-sandbox-python"

    def test_get_builtin_manifest_with_version(self):
        """Test getting builtin manifest with version tag."""
        manifest = get_builtin_manifest("llmcore-sandbox-python:1.0.0")
        assert manifest is not None
        assert manifest.name == "llmcore-sandbox-python"

    def test_get_builtin_manifest_unknown(self):
        """Test that unknown manifest returns None."""
        manifest = get_builtin_manifest("unknown-image")
        assert manifest is None

    def test_list_builtin_manifests(self):
        """Test listing all builtin manifests."""
        manifests = list_builtin_manifests()
        assert len(manifests) >= 6  # At least 6 builtin images

        # Check expected images
        assert "llmcore-sandbox-base" in manifests
        assert "llmcore-sandbox-python" in manifests
        assert "llmcore-sandbox-nodejs" in manifests

    def test_list_builtin_returns_copy(self):
        """Test that list_builtin_manifests returns a copy."""
        manifests1 = list_builtin_manifests()
        manifests2 = list_builtin_manifests()
        assert manifests1 is not manifests2


# ==============================================================================
# Constants Tests
# ==============================================================================


class TestManifestConstants:
    """Tests for manifest module constants."""

    def test_manifest_path_in_container(self):
        """Test the in-container manifest path constant."""
        assert MANIFEST_PATH_IN_CONTAINER == "/etc/llmcore/capabilities.json"
