#!/bin/bash
# Fix incorrect postgres_storage imports in manager.py

set -e

echo "🔧 Fixing incorrect imports in storage/manager.py"
echo ""

MANAGER_FILE="src/llmcore/storage/manager.py"

if [ ! -f "$MANAGER_FILE" ]; then
    echo "❌ Error: $MANAGER_FILE not found"
    exit 1
fi

echo "📝 Backing up original file..."
cp "$MANAGER_FILE" "${MANAGER_FILE}.backup"

echo "🔄 Replacing incorrect imports..."

# Replace the incorrect imports
sed -i 's/from \.postgres_storage import PgVectorStorage/from .pgvector_storage import PgVectorStorage/g' "$MANAGER_FILE"
sed -i 's/from \.postgres_storage import PostgresSessionStorage/from .postgres_session_storage import PostgresSessionStorage/g' "$MANAGER_FILE"

echo "✅ Import fixes applied"
echo ""
echo "Changes made:"
echo "  - .postgres_storage.PgVectorStorage → .pgvector_storage.PgVectorStorage"
echo "  - .postgres_storage.PostgresSessionStorage → .postgres_session_storage.PostgresSessionStorage"
echo ""
echo "Backup saved as: ${MANAGER_FILE}.backup"
