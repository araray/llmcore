#!/bin/bash
# Run this in your llmcore project directory

echo "📋 Checking pyproject.toml for console_scripts..."
echo ""

# Check if console_scripts section exists and what's in it
if grep -A 10 "\[project.scripts\]" pyproject.toml 2>/dev/null; then
    echo "⚠️  Found [project.scripts] section - checking contents..."
    # Check if any api_server or task_master scripts remain
    if grep -E "(api_server|task_master|llmcore-server|llmcore-worker)" pyproject.toml; then
        echo "❌ FAIL: Service-related console scripts still present"
        echo "   Action needed: Remove these from pyproject.toml"
    else
        echo "✅ PASS: No service-related console scripts found"
    fi
else
    echo "✅ PASS: No [project.scripts] section (expected for pure library)"
fi

echo ""
echo "🐍 Running static compilation check..."
echo ""

if python3 -m compileall src/llmcore -q; then
    echo "✅ PASS: All Python files compile successfully"
else
    echo "❌ FAIL: Compilation errors detected"
    echo "   Run: python3 -m compileall src/llmcore"
    echo "   to see detailed errors"
fi

echo ""
echo "📦 Checking pyproject.toml dependencies..."
echo ""

# Service dependencies that should be removed
SERVICE_DEPS=("fastapi" "uvicorn" "arq" "redis" "bcrypt" "asyncpg" "prometheus-client" "prometheus-fastapi-instrumentator" "opentelemetry-distro" "python-multipart")

DEPS_TO_REMOVE=()
for dep in "${SERVICE_DEPS[@]}"; do
    if grep -q "\"$dep" pyproject.toml; then
        DEPS_TO_REMOVE+=("$dep")
    fi
done

if [ ${#DEPS_TO_REMOVE[@]} -eq 0 ]; then
    echo "✅ PASS: All service dependencies removed from pyproject.toml"
else
    echo "⚠️  WARNING: Found service-related dependencies:"
    for dep in "${DEPS_TO_REMOVE[@]}"; do
        echo "   - $dep"
    done
    echo ""
    echo "   These can be removed unless needed for other purposes"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "VERIFICATION COMPLETE"
echo "═══════════════════════════════════════════════════"
