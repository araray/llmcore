#!/bin/bash
# Step 1.1 - Delete service-oriented components from llmcore
# This script removes api_server, task_master, and migrations directories

set -e  # Exit on error

echo "🔍 Step 1.1: Surgical Excision of Service Components"
echo "=================================================="
echo ""

# Define the repository root (adjust if needed)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "📍 Working directory: $REPO_ROOT"
echo ""

# Function to safely remove directory
remove_directory() {
    local dir_path="$1"
    local dir_name="$2"
    
    if [ -d "$dir_path" ]; then
        echo "🗑️  Removing $dir_name..."
        rm -rf "$dir_path"
        echo "✅ $dir_name removed"
    else
        echo "⚠️  $dir_name not found (already removed?)"
    fi
    echo ""
}

# Remove the three main directories
echo "🎯 Removing service-oriented directories..."
echo ""

remove_directory "src/llmcore/api_server" "api_server/"
remove_directory "src/llmcore/task_master" "task_master/"
remove_directory "migrations" "migrations/"

echo "✅ Directory deletions complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Update pyproject.toml (see provided full file)"
echo "   2. Update src/llmcore/api.py (see provided full file)"
echo "   3. Update src/llmcore/__init__.py (see provided full file)"
echo "   4. Run: python -m compileall src/llmcore"
echo "   5. Run: pip install -e ."
echo ""
