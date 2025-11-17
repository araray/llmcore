#!/usr/bin/env python3
"""
Step 1.1 Cleanup Script - Surgical Excision of Service Components

This script safely removes service-oriented components from llmcore:
- api_server/ directory
- task_master/ directory  
- migrations/ directory
- All import references to these components
- Tests for these components

It generates a detailed report of all changes.
"""

import os
import sys
import shutil
import re
from pathlib import Path
from typing import List, Tuple, Dict, Set

# ANSI color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


class CleanupReport:
    """Tracks all cleanup operations for final report."""
    
    def __init__(self):
        self.deleted_dirs: List[str] = []
        self.deleted_files: List[str] = []
        self.modified_files: List[Tuple[str, int]] = []  # (file, lines_changed)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def add_deleted_dir(self, path: str):
        self.deleted_dirs.append(path)
        
    def add_deleted_file(self, path: str):
        self.deleted_files.append(path)
        
    def add_modified_file(self, path: str, lines_changed: int):
        self.modified_files.append((path, lines_changed))
        
    def add_error(self, msg: str):
        self.errors.append(msg)
        
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        
    def print_report(self):
        """Print comprehensive cleanup report."""
        print(f"\n{'='*70}")
        print(f"{BLUE}STEP 1.1 CLEANUP REPORT{NC}")
        print(f"{'='*70}\n")
        
        # Deleted directories
        if self.deleted_dirs:
            print(f"{GREEN}✅ Deleted Directories ({len(self.deleted_dirs)}):{NC}")
            for d in self.deleted_dirs:
                print(f"   🗑️  {d}")
            print()
        
        # Deleted files
        if self.deleted_files:
            print(f"{GREEN}✅ Deleted Test Files ({len(self.deleted_files)}):{NC}")
            for f in self.deleted_files:
                print(f"   🗑️  {f}")
            print()
        
        # Modified files
        if self.modified_files:
            print(f"{GREEN}✅ Modified Files ({len(self.modified_files)}):{NC}")
            for f, lines in self.modified_files:
                print(f"   ✏️  {f} ({lines} lines changed)")
            print()
        
        # Warnings
        if self.warnings:
            print(f"{YELLOW}⚠️  Warnings ({len(self.warnings)}):{NC}")
            for w in self.warnings:
                print(f"   • {w}")
            print()
        
        # Errors
        if self.errors:
            print(f"{RED}❌ Errors ({len(self.errors)}):{NC}")
            for e in self.errors:
                print(f"   • {e}")
            print()
        
        # Summary
        print(f"{'='*70}")
        total_changes = len(self.deleted_dirs) + len(self.deleted_files) + len(self.modified_files)
        if self.errors:
            print(f"{RED}❌ Cleanup completed with {len(self.errors)} error(s){NC}")
            return False
        else:
            print(f"{GREEN}✅ Cleanup completed successfully! ({total_changes} total changes){NC}")
            return True


def find_project_root() -> Path:
    """Find the llmcore project root directory."""
    current = Path.cwd()
    
    # Look for pyproject.toml or src/llmcore
    for _ in range(5):  # Check up to 5 levels up
        if (current / 'pyproject.toml').exists() and (current / 'src' / 'llmcore').exists():
            return current
        current = current.parent
    
    raise FileNotFoundError("Could not find llmcore project root (looking for src/llmcore)")


def safe_delete_directory(path: Path, report: CleanupReport) -> bool:
    """Safely delete a directory with confirmation."""
    if not path.exists():
        report.add_warning(f"Directory not found (already deleted?): {path}")
        return True
    
    if not path.is_dir():
        report.add_error(f"Path exists but is not a directory: {path}")
        return False
    
    try:
        # Count files for user information
        file_count = sum(1 for _ in path.rglob('*') if _.is_file())
        
        print(f"{YELLOW}🗑️  Deleting: {path} ({file_count} files){NC}")
        shutil.rmtree(path)
        report.add_deleted_dir(str(path.relative_to(find_project_root())))
        print(f"{GREEN}   ✅ Deleted successfully{NC}")
        return True
        
    except Exception as e:
        report.add_error(f"Failed to delete {path}: {e}")
        return False


def clean_imports_from_file(file_path: Path, patterns: List[str], report: CleanupReport) -> bool:
    """Remove import lines matching patterns from a file."""
    if not file_path.exists():
        report.add_warning(f"File not found: {file_path}")
        return True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        # Filter out lines matching any pattern
        new_lines = []
        removed_count = 0
        
        for line in original_lines:
            should_remove = False
            for pattern in patterns:
                if re.search(pattern, line):
                    should_remove = True
                    removed_count += 1
                    break
            
            if not should_remove:
                new_lines.append(line)
        
        if removed_count > 0:
            # Write back the cleaned file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            rel_path = file_path.relative_to(find_project_root())
            report.add_modified_file(str(rel_path), removed_count)
            print(f"{GREEN}   ✏️  Cleaned {removed_count} line(s) from: {rel_path}{NC}")
        
        return True
        
    except Exception as e:
        report.add_error(f"Failed to clean imports from {file_path}: {e}")
        return False


def scan_for_references(root: Path, patterns: List[str]) -> Dict[str, List[str]]:
    """Scan Python files for references to deleted components."""
    references = {}
    
    for py_file in root.rglob('*.py'):
        # Skip deleted directories
        if any(part in ['api_server', 'task_master', 'migrations'] for part in py_file.parts):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_refs = []
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE):
                    file_refs.append(pattern)
            
            if file_refs:
                references[str(py_file.relative_to(root))] = file_refs
                
        except Exception:
            pass  # Skip files that can't be read
    
    return references


def main():
    """Main cleanup procedure."""
    print(f"\n{BLUE}{'='*70}")
    print("STEP 1.1: Surgical Excision of Service Components")
    print(f"{'='*70}{NC}\n")
    
    report = CleanupReport()
    
    try:
        project_root = find_project_root()
        print(f"{GREEN}📍 Project root: {project_root}{NC}\n")
    except FileNotFoundError as e:
        print(f"{RED}❌ Error: {e}{NC}")
        sys.exit(1)
    
    # Change to project root
    os.chdir(project_root)
    
    # ========================================================================
    # PHASE 1: Delete Service Component Directories
    # ========================================================================
    print(f"{BLUE}PHASE 1: Deleting Service Component Directories{NC}\n")
    
    directories_to_delete = [
        project_root / 'src' / 'llmcore' / 'api_server',
        project_root / 'src' / 'llmcore' / 'task_master',
        project_root / 'migrations',
    ]
    
    for directory in directories_to_delete:
        safe_delete_directory(directory, report)
    
    # ========================================================================
    # PHASE 2: Delete Test Files for Deleted Components
    # ========================================================================
    print(f"\n{BLUE}PHASE 2: Deleting Test Files{NC}\n")
    
    test_dirs_to_delete = [
        project_root / 'tests' / 'api_server',
    ]
    
    for test_dir in test_dirs_to_delete:
        if test_dir.exists():
            safe_delete_directory(test_dir, report)
    
    # Delete individual test files
    test_files_to_delete = [
        project_root / 'tests' / 'test_task_master.py',
    ]
    
    for test_file in test_files_to_delete:
        if test_file.exists():
            print(f"{YELLOW}🗑️  Deleting test file: {test_file.name}{NC}")
            test_file.unlink()
            report.add_deleted_file(str(test_file.relative_to(project_root)))
            print(f"{GREEN}   ✅ Deleted{NC}")
    
    # ========================================================================
    # PHASE 3: Clean Import References
    # ========================================================================
    print(f"\n{BLUE}PHASE 3: Cleaning Import References{NC}\n")
    
    # Patterns to search for and remove
    import_patterns = [
        r'from\s+\.?\.?api_server\s+import',
        r'from\s+\.?\.?task_master\s+import',
        r'from\s+llmcore\.api_server',
        r'from\s+llmcore\.task_master',
        r'import\s+llmcore\.api_server',
        r'import\s+llmcore\.task_master',
    ]
    
    # Files to clean
    files_to_clean = [
        project_root / 'src' / 'llmcore' / '__init__.py',
    ]
    
    for file_path in files_to_clean:
        clean_imports_from_file(file_path, import_patterns, report)
    
    # ========================================================================
    # PHASE 4: Scan for Remaining References
    # ========================================================================
    print(f"\n{BLUE}PHASE 4: Scanning for Remaining References{NC}\n")
    
    scan_patterns = [
        r'api_server\.',
        r'task_master\.',
        r'from\s+\.api_server',
        r'from\s+\.task_master',
    ]
    
    src_dir = project_root / 'src' / 'llmcore'
    references = scan_for_references(src_dir, scan_patterns)
    
    if references:
        print(f"{YELLOW}⚠️  Found remaining references in {len(references)} file(s):{NC}")
        for file_path, patterns in references.items():
            print(f"   📄 {file_path}")
            for pattern in patterns:
                print(f"      • {pattern}")
            report.add_warning(f"Remaining reference in {file_path}: {patterns}")
    else:
        print(f"{GREEN}✅ No remaining references found{NC}")
    
    # ========================================================================
    # PHASE 5: Run Static Checks
    # ========================================================================
    print(f"\n{BLUE}PHASE 5: Running Static Checks{NC}\n")
    
    print(f"🐍 Compiling Python files...")
    import py_compile
    import compileall
    
    try:
        # Compile all Python files in src/llmcore
        compileall.compile_dir(
            str(src_dir),
            quiet=1,
            force=True,
            legacy=True
        )
        print(f"{GREEN}✅ All Python files compile successfully{NC}")
    except Exception as e:
        report.add_error(f"Compilation failed: {e}")
        print(f"{RED}❌ Compilation errors detected{NC}")
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    success = report.print_report()
    
    if success:
        print(f"\n{GREEN}{'='*70}")
        print("NEXT STEPS:")
        print(f"{'='*70}{NC}")
        print("""
1. Review the changes above
2. Apply the provided fixes:
   - StorageManager.close() method
   - Updated __init__.py (if needed)
3. Run: pip install -e .
4. Run: python scripts/test_library_mode.py
5. Commit changes with provided commit message
        """)
        sys.exit(0)
    else:
        print(f"\n{RED}⚠️  Please review errors above before proceeding{NC}")
        sys.exit(1)


if __name__ == '__main__':
    main()
