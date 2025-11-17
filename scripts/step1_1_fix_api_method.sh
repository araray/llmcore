#!/bin/bash
# Fix incorrect method call in api.py

set -e

echo "🔧 Fixing method name in src/llmcore/api.py"
echo ""

API_FILE="src/llmcore/api.py"

if [ ! -f "$API_FILE" ]; then
    echo "❌ Error: $API_FILE not found"
    exit 1
fi

echo "📝 Backing up original file..."
cp "$API_FILE" "${API_FILE}.backup_method"

echo "🔄 Replacing incorrect method call..."

# Replace the incorrect method call
sed -i 's/return self\._provider_manager\.list_providers()/return self._provider_manager.get_available_providers()/g' "$API_FILE"

echo "✅ Method name fix applied"
echo ""
echo "Change made:"
echo "  - self._provider_manager.list_providers()"
echo "  → self._provider_manager.get_available_providers()"
echo ""
echo "Backup saved as: ${API_FILE}.backup_method"
