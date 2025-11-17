#!/usr/bin/env python3
"""
Step 1.1 Final Fixes Script

This script applies all remaining fixes to complete Step 1.1:
1. Add StorageManager.close() method
2. Add ProviderManager.close_all() method  
3. Remove api_server references from agent files

Usage: python scripts/step1_1_apply_fixes.py
"""

import os
import sys
import re
from pathlib import Path

def find_project_root():
    """Find llmcore project root."""
    current = Path.cwd()
    for _ in range(5):
        if (current / 'pyproject.toml').exists() and (current / 'src' / 'llmcore').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find llmcore project root")

def apply_storage_manager_fix(project_root):
    """Add close() method to StorageManager."""
    print("\n🔧 Fix 1: Adding StorageManager.close() method...")
    
    file_path = project_root / 'src' / 'llmcore' / 'storage' / 'manager.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'async def close(self)' in content and 'await self.close_storages()' in content:
        print("   ✅ Already fixed - close() method exists")
        return True
    
    # Find the close_storages method and add close() after it
    pattern = r'(async def close_storages\(self\) -> None:.*?logger\.info\("Storage manager cleanup complete.*?"\))'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("   ❌ Could not find insertion point")
        return False
    
    insertion_point = match.end()
    
    new_method = '''

    async def close(self) -> None:
        """
        Alias for close_storages() to maintain API compatibility.
        
        LLMCore.close() calls StorageManager.close(), so this method
        delegates to close_storages() which does the actual cleanup.
        """
        await self.close_storages()
'''
    
    new_content = content[:insertion_point] + new_method + content[insertion_point:]
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("   ✅ StorageManager.close() method added")
    return True

def apply_provider_manager_fix(project_root):
    """Add close_all() method to ProviderManager."""
    print("\n🔧 Fix 2: Adding ProviderManager.close_all() method...")
    
    file_path = project_root / 'src' / 'llmcore' / 'providers' / 'manager.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'async def close_all(self)' in content:
        print("   ✅ Already fixed - close_all() method exists")
        return True
    
    # Find the close_providers method and add close_all() after it
    pattern = r'(async def close_providers\(self\) -> None:.*?logger\.info\("Provider connections closure attempt complete\."\))'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("   ❌ Could not find insertion point")
        return False
    
    insertion_point = match.end()
    
    new_method = '''

    async def close_all(self) -> None:
        """
        Alias for close_providers() to maintain API compatibility.
        
        LLMCore.close() calls ProviderManager.close_all(), so this method
        delegates to close_providers() which does the actual cleanup.
        """
        await self.close_providers()
'''
    
    new_content = content[:insertion_point] + new_method + content[insertion_point:]
    
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print("   ✅ ProviderManager.close_all() method added")
    return True

def remove_api_server_references(project_root):
    """Remove api_server references from agent files."""
    print("\n🔧 Fix 3: Removing api_server references...")
    
    files_to_fix = [
        'src/llmcore/agents/manager.py',
        'src/llmcore/agents/cognitive_cycle.py',
        'src/llmcore/providers/base.py'
    ]
    
    patterns_to_remove = [
        # Import statements
        r'from\s+\.\.api_server\s+import.*?\n',
        r'from\s+\.\.api_server\..*?import.*?\n',
        r'import\s+.*?\.api_server\..*?\n',
        
        # Try-except blocks importing from api_server
        r'\s*try:\s*\n\s*from\s+\.\.api_server\..*?import.*?\n.*?except.*?:\n.*?\n',
        
        # Specific patterns found in the code
        r'\s*from\s+\.\.api_server\.metrics\s+import\s+record_agent_execution\n',
        r'\s*from\s+\.\.api_server\.middleware\.observability\s+import\s+get_current_request_context\n',
        
        # Usage of api_server functions (the actual calls)
        r'\s*try:\s*\n\s*from\s+\.\.api_server\.metrics.*?except.*?:\n.*?logger\.debug.*?\n',
        r'\s*record_agent_execution\([^)]*\)\s*\n',
        r'\s*context\s*=\s*get_current_request_context\(\)\s*\n',
        r'\s*tenant_id\s*=\s*context\.get\([^)]*\)\s*\n',
    ]
    
    fixed_count = 0
    
    for file_rel_path in files_to_fix:
        file_path = project_root / file_rel_path
        
        if not file_path.exists():
            print(f"   ⚠️  File not found: {file_rel_path}")
            continue
        
        with open(file_path, 'r') as f:
            original_content = f.read()
        
        content = original_content
        lines_removed = 0
        
        # Apply all patterns
        for pattern in patterns_to_remove:
            matches = list(re.finditer(pattern, content))
            if matches:
                lines_removed += len(matches)
                content = re.sub(pattern, '', content)
        
        # Additional cleanup: remove empty try-except blocks that might remain
        content = re.sub(r'\s*try:\s*\n\s*except\s+Exception\s+as\s+\w+:\s*\n\s*logger\.debug\([^)]*\)\s*\n', '', content)
        
        # Remove duplicate blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   ✅ Cleaned {file_rel_path} ({lines_removed} references removed)")
            fixed_count += 1
        else:
            print(f"   ℹ️  No changes needed in {file_rel_path}")
    
    return fixed_count > 0

def main():
    """Main execution."""
    print("="*70)
    print("STEP 1.1 FINAL FIXES")
    print("="*70)
    
    try:
        project_root = find_project_root()
        print(f"📍 Project root: {project_root}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    os.chdir(project_root)
    
    success_count = 0
    total_fixes = 3
    
    # Apply all fixes
    if apply_storage_manager_fix(project_root):
        success_count += 1
    
    if apply_provider_manager_fix(project_root):
        success_count += 1
    
    if remove_api_server_references(project_root):
        success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("FIXES APPLIED SUMMARY")
    print("="*70)
    print(f"✅ Successfully applied {success_count}/{total_fixes} fixes")
    
    if success_count == total_fixes:
        print("\n🎉 All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Run: pip install -e .")
        print("2. Run: python scripts/test_library_mode.py")
        print("3. Run: python scripts/test_session_management.py")
        print("4. Verify no errors in output")
        sys.exit(0)
    else:
        print("\n⚠️  Some fixes could not be applied automatically")
        print("Please review the output above and apply remaining fixes manually")
        sys.exit(1)

if __name__ == '__main__':
    main()
