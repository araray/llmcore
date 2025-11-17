#!/usr/bin/env python3
"""
Step 1.3 Automated Patch Applicator
====================================

This script automatically applies all code changes for Step 1.3 refactoring:
- Updates src/llmcore/api.py (3 changes)
- Updates src/llmcore/memory/manager.py (2 changes)
- Creates backups before any changes
- Validates changes were applied correctly
- Provides rollback capability

Usage:
    python apply_step_1_3_patches.py [--dry-run] [--rollback]

Options:
    --dry-run    Show what would be changed without making changes
    --rollback   Restore files from backups
    --help       Show this help message

Requirements:
    - Run from the root of your llmcore project directory
    - Python 3.7+
    - No external dependencies required
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PatchApplicator:
    """Handles the application of Step 1.3 patches."""
    
    def __init__(self, root_dir: Path, dry_run: bool = False):
        """
        Initialize the patch applicator.
        
        Args:
            root_dir: Root directory of the llmcore project
            dry_run: If True, only show what would be changed
        """
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.backup_suffix = f".backup_step1.3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Files to patch
        self.files_to_patch = {
            'api': self.root_dir / 'src' / 'llmcore' / 'api.py',
            'memory': self.root_dir / 'src' / 'llmcore' / 'memory' / 'manager.py',
        }
        
        # Track changes
        self.changes_made = []
        self.backups_created = []
        
    def print_header(self, message: str):
        """Print a formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
        print(f"{message}")
        print(f"{'='*70}{Colors.ENDC}\n")
    
    def print_success(self, message: str):
        """Print a success message."""
        print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")
    
    def print_error(self, message: str):
        """Print an error message."""
        print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")
    
    def print_info(self, message: str):
        """Print an info message."""
        print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")
    
    def validate_environment(self) -> bool:
        """
        Validate that we're in the correct directory with required files.
        
        Returns:
            True if environment is valid, False otherwise
        """
        self.print_header("Step 1: Validating Environment")
        
        # Check if we're in a llmcore project
        if not (self.root_dir / 'src' / 'llmcore').exists():
            self.print_error("Not in a llmcore project directory!")
            self.print_info("Please run this script from the root of your llmcore project.")
            return False
        
        self.print_success(f"Found llmcore project at: {self.root_dir}")
        
        # Check if required files exist
        missing_files = []
        for name, path in self.files_to_patch.items():
            if not path.exists():
                missing_files.append(str(path))
            else:
                self.print_success(f"Found {name}: {path.relative_to(self.root_dir)}")
        
        if missing_files:
            self.print_error("Missing required files:")
            for f in missing_files:
                print(f"  - {f}")
            return False
        
        return True
    
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of the file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to backup file, or None if backup failed
        """
        backup_path = file_path.with_suffix(file_path.suffix + self.backup_suffix)
        
        try:
            if self.dry_run:
                self.print_info(f"[DRY RUN] Would create backup: {backup_path.name}")
                return backup_path
            
            shutil.copy2(file_path, backup_path)
            self.backups_created.append(backup_path)
            self.print_success(f"Created backup: {backup_path.name}")
            return backup_path
            
        except Exception as e:
            self.print_error(f"Failed to create backup: {e}")
            return None
    
    def apply_patch(self, file_path: Path, search: str, replace: str, 
                   description: str, expect_count: int = 1) -> bool:
        """
        Apply a single patch to a file.
        
        Args:
            file_path: Path to file to patch
            search: String to search for
            replace: String to replace with
            description: Description of the change
            expect_count: Expected number of replacements
            
        Returns:
            True if patch applied successfully, False otherwise
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count occurrences
            count = content.count(search)
            
            if count == 0:
                self.print_warning(f"Pattern not found (may already be patched): {description}")
                return True  # Consider this success - already patched
            
            if count != expect_count:
                self.print_warning(
                    f"Found {count} occurrences, expected {expect_count}: {description}"
                )
                response = input("Continue anyway? (y/n): ").lower()
                if response != 'y':
                    return False
            
            # Apply replacement
            new_content = content.replace(search, replace)
            
            if self.dry_run:
                self.print_info(f"[DRY RUN] Would apply: {description}")
                self.print_info(f"  Found {count} occurrence(s)")
                return True
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.print_success(f"Applied: {description}")
            self.changes_made.append({
                'file': file_path,
                'description': description,
                'count': count
            })
            
            return True
            
        except Exception as e:
            self.print_error(f"Failed to apply patch: {e}")
            return False
    
    def patch_api_file(self) -> bool:
        """
        Apply all patches to api.py.
        
        Returns:
            True if all patches applied successfully
        """
        self.print_header("Step 3: Patching src/llmcore/api.py")
        
        api_file = self.files_to_patch['api']
        
        # Create backup
        if not self.create_backup(api_file):
            return False
        
        patches = [
            # Patch 1: SessionManager initialization
            {
                'search': 'self._session_manager = SessionManager(self._storage_manager.get_session_storage())',
                'replace': 'self._session_manager = SessionManager(self._storage_manager.session_storage)',
                'description': 'Update SessionManager initialization',
                'expect': 1
            },
            # Patch 2: add_documents_to_vector_store
            {
                'search': 'vector_storage = self._storage_manager.get_vector_storage(collection_name)',
                'replace': 'vector_storage = self._storage_manager.vector_storage',
                'description': 'Update add_documents_to_vector_store method',
                'expect': 1
            },
            # Patch 3: search_vector_store
            {
                'search': 'vector_storage = self._storage_manager.get_vector_storage(collection_name)\n        return await vector_storage.search(query, k=k, metadata_filter=metadata_filter)',
                'replace': 'vector_storage = self._storage_manager.vector_storage\n        return await vector_storage.search(query, k=k, metadata_filter=metadata_filter)',
                'description': 'Update search_vector_store method',
                'expect': 1
            }
        ]
        
        success = True
        for patch in patches:
            if not self.apply_patch(
                api_file,
                patch['search'],
                patch['replace'],
                patch['description'],
                patch['expect']
            ):
                success = False
                break
        
        return success
    
    def patch_memory_file(self) -> bool:
        """
        Apply all patches to memory/manager.py.
        
        Returns:
            True if all patches applied successfully
        """
        self.print_header("Step 4: Patching src/llmcore/memory/manager.py")
        
        memory_file = self.files_to_patch['memory']
        
        # Create backup
        if not self.create_backup(memory_file):
            return False
        
        patches = [
            # Patch 1: prepare_context method
            {
                'search': 'vector_storage = self._storage_manager.get_vector_storage()',
                'replace': 'vector_storage = self._storage_manager.vector_storage',
                'description': 'Update prepare_context RAG retrieval',
                'expect': 2  # Should appear twice in the file
            }
        ]
        
        success = True
        for patch in patches:
            if not self.apply_patch(
                memory_file,
                patch['search'],
                patch['replace'],
                patch['description'],
                patch['expect']
            ):
                success = False
                break
        
        return success
    
    def verify_changes(self) -> bool:
        """
        Verify that all changes were applied correctly.
        
        Returns:
            True if verification passed
        """
        self.print_header("Step 5: Verifying Changes")
        
        if self.dry_run:
            self.print_info("[DRY RUN] Skipping verification")
            return True
        
        verification_passed = True
        
        # Check that old patterns are gone
        old_patterns = [
            'get_session_storage()',
            'get_vector_storage()',
        ]
        
        for name, file_path in self.files_to_patch.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in old_patterns:
                if pattern in content:
                    self.print_warning(
                        f"Old pattern still found in {name}: '{pattern}'"
                    )
                    verification_passed = False
        
        # Check that new patterns exist
        new_patterns = [
            '.session_storage',
            '.vector_storage',
        ]
        
        api_file = self.files_to_patch['api']
        with open(api_file, 'r', encoding='utf-8') as f:
            api_content = f.read()
        
        for pattern in new_patterns:
            if pattern not in api_content:
                self.print_warning(
                    f"New pattern not found in api.py: '{pattern}'"
                )
                verification_passed = False
        
        if verification_passed:
            self.print_success("All verification checks passed!")
        else:
            self.print_warning("Some verification checks failed. Review changes manually.")
        
        return verification_passed
    
    def print_summary(self):
        """Print a summary of changes made."""
        self.print_header("Summary")
        
        if self.dry_run:
            print(f"{Colors.OKCYAN}DRY RUN MODE - No actual changes were made{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}Backups Created:{Colors.ENDC}")
        if self.backups_created:
            for backup in self.backups_created:
                print(f"  - {backup.relative_to(self.root_dir)}")
        else:
            print("  (none - dry run mode)")
        
        print(f"\n{Colors.BOLD}Changes Applied:{Colors.ENDC}")
        if self.changes_made:
            for change in self.changes_made:
                file_rel = change['file'].relative_to(self.root_dir)
                print(f"  - {file_rel}")
                print(f"    {change['description']} ({change['count']} occurrence(s))")
        else:
            print("  (none - dry run mode)")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("  1. Review the changes in your editor")
        print("  2. Run tests: pytest tests/")
        print("  3. If tests pass, commit changes")
        print("  4. If issues occur, run: python apply_step_1_3_patches.py --rollback")
        
        if not self.dry_run:
            print(f"\n{Colors.OKGREEN}✓ Step 1.3 patches applied successfully!{Colors.ENDC}")
    
    def rollback(self) -> bool:
        """
        Rollback changes by restoring from backups.
        
        Returns:
            True if rollback successful
        """
        self.print_header("Rolling Back Changes")
        
        # Find backup files
        backup_pattern = f"*{self.backup_suffix}"
        backup_files = []
        
        for file_path in self.files_to_patch.values():
            parent_dir = file_path.parent
            for backup in parent_dir.glob(backup_pattern):
                backup_files.append(backup)
        
        if not backup_files:
            # Try to find any Step 1.3 backups
            for file_path in self.files_to_patch.values():
                parent_dir = file_path.parent
                for backup in parent_dir.glob("*.backup_step1.3_*"):
                    backup_files.append(backup)
        
        if not backup_files:
            self.print_error("No backup files found!")
            self.print_info("Backup files should match pattern: *.backup_step1.3_*")
            return False
        
        print(f"Found {len(backup_files)} backup file(s):")
        for backup in backup_files:
            print(f"  - {backup.name}")
        
        response = input("\nRestore from these backups? (y/n): ").lower()
        if response != 'y':
            print("Rollback cancelled.")
            return False
        
        success = True
        for backup in backup_files:
            # Determine original file path
            original = backup.with_suffix('')
            for _ in range(10):  # Remove multiple suffixes if needed
                if not original.suffix.startswith('.backup'):
                    break
                original = original.with_suffix('')
            
            try:
                shutil.copy2(backup, original)
                self.print_success(f"Restored: {original.name}")
            except Exception as e:
                self.print_error(f"Failed to restore {original.name}: {e}")
                success = False
        
        if success:
            self.print_success("\nRollback completed successfully!")
            print("\nBackup files preserved. You can delete them manually if desired:")
            for backup in backup_files:
                print(f"  rm {backup}")
        
        return success
    
    def run(self) -> bool:
        """
        Run the complete patch application process.
        
        Returns:
            True if all patches applied successfully
        """
        # Validate environment
        if not self.validate_environment():
            return False
        
        if self.dry_run:
            self.print_header("DRY RUN MODE - No changes will be made")
        
        # Apply patches
        self.print_header("Step 2: Creating Backups")
        
        if not self.patch_api_file():
            self.print_error("Failed to patch api.py")
            return False
        
        if not self.patch_memory_file():
            self.print_error("Failed to patch memory/manager.py")
            return False
        
        # Verify changes
        if not self.verify_changes():
            self.print_warning("Verification had warnings, but patches were applied")
        
        # Print summary
        self.print_summary()
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Apply Step 1.3 patches to llmcore project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply patches (with confirmation)
  python apply_step_1_3_patches.py
  
  # Preview changes without applying
  python apply_step_1_3_patches.py --dry-run
  
  # Rollback changes
  python apply_step_1_3_patches.py --rollback
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Restore files from backups'
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    root_dir = Path.cwd()
    
    # Create applicator
    applicator = PatchApplicator(root_dir, dry_run=args.dry_run)
    
    try:
        if args.rollback:
            success = applicator.rollback()
        else:
            success = applicator.run()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
