#!/bin/bash
# Step 1.1 - Verification script for service component removal
# This script checks that all service components are removed and core library still works

set -e  # Exit on error

echo "🔍 Step 1.1 Verification: Service Components Removal"
echo "===================================================="
echo ""

# Define the repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "📍 Working directory: $REPO_ROOT"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if directory exists
check_directory_removed() {
    local dir_path="$1"
    local dir_name="$2"
    
    if [ -d "$dir_path" ]; then
        echo -e "${RED}❌ FAIL: $dir_name still exists${NC}"
        return 1
    else
        echo -e "${GREEN}✅ PASS: $dir_name successfully removed${NC}"
        return 0
    fi
}

# Function to check if import is removed from a file
check_import_removed() {
    local file_path="$1"
    local import_pattern="$2"
    local description="$3"
    
    if [ ! -f "$file_path" ]; then
        echo -e "${YELLOW}⚠️  SKIP: $file_path not found${NC}"
        return 0
    fi
    
    if grep -q "$import_pattern" "$file_path"; then
        echo -e "${RED}❌ FAIL: $description still present in $file_path${NC}"
        return 1
    else
        echo -e "${GREEN}✅ PASS: $description removed from $file_path${NC}"
        return 0
    fi
}

# Function to check if dependency is removed from pyproject.toml
check_dependency_removed() {
    local dependency_name="$1"
    
    if grep -q "\"$dependency_name" pyproject.toml; then
        echo -e "${RED}❌ FAIL: $dependency_name still in pyproject.toml${NC}"
        return 1
    else
        echo -e "${GREEN}✅ PASS: $dependency_name removed from pyproject.toml${NC}"
        return 0
    fi
}

echo "🗂️  Checking directory removals..."
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Check directories
if check_directory_removed "src/llmcore/api_server" "api_server/"; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

if check_directory_removed "src/llmcore/task_master" "task_master/"; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

if check_directory_removed "migrations" "migrations/"; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

echo ""
echo "📦 Checking dependency removals in pyproject.toml..."
echo ""

# Check service dependencies removed
for dep in "fastapi" "uvicorn" "arq" "redis" "bcrypt" "asyncpg" "prometheus-client" "prometheus-fastapi-instrumentator" "opentelemetry-distro" "python-multipart"; do
    if check_dependency_removed "$dep"; then
        ((PASS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "🔍 Checking import removals..."
echo ""

# Check that AgentManager is NOT initialized in api.py
if check_import_removed "src/llmcore/api.py" "self._agent_manager = AgentManager" "AgentManager initialization"; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

# Check that get_agent_manager method is removed
if check_import_removed "src/llmcore/api.py" "def get_agent_manager" "get_agent_manager method"; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

echo ""
echo "🐍 Running Python compilation check..."
echo ""

if python3 -m compileall src/llmcore -q; then
    echo -e "${GREEN}✅ PASS: All Python files compile successfully${NC}"
    ((PASS_COUNT++))
else
    echo -e "${RED}❌ FAIL: Python compilation errors detected${NC}"
    ((FAIL_COUNT++))
fi

echo ""
echo "📊 Running import test..."
echo ""

# Test that LLMCore can be imported
python3 << 'EOF'
import sys
try:
    from llmcore import LLMCore
    print("\033[0;32m✅ PASS: LLMCore can be imported\033[0m")
    sys.exit(0)
except ImportError as e:
    print(f"\033[0;31m❌ FAIL: Cannot import LLMCore: {e}\033[0m")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    ((PASS_COUNT++))
else
    ((FAIL_COUNT++))
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "📊 Verification Summary"
echo "═══════════════════════════════════════════════════"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Step 1.1 complete.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Commit these changes"
    echo "  2. Run: pip install -e ."
    echo "  3. Test basic LLMCore functionality"
    echo "  4. Proceed to Step 1.2 (if applicable)"
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed. Please review and fix before proceeding.${NC}"
    exit 1
fi
