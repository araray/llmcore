#!/usr/bin/env python3
"""
Add missing methods to SessionManager.
"""

import sys

# Read the file
with open('src/llmcore/sessions/manager.py', 'r') as f:
    content = f.read()

# Check if the method already exists
if 'async def list_sessions' in content:
    print("✅ list_sessions method already exists in SessionManager")
    sys.exit(0)

# Find the position to insert the new method (after load_or_create_session)
insert_marker = '        logger.debug(f"Session \'{session.id}\' saved via SessionManager.")'

if insert_marker not in content:
    print("❌ Could not find insertion point in SessionManager")
    sys.exit(1)

# Define the new methods to add
new_methods = '''

    async def list_sessions(self, limit: Optional[int] = None) -> list:
        """
        Lists all available chat sessions with metadata.

        Args:
            limit: Optional maximum number of sessions to return (not supported by all backends)

        Returns:
            List of session metadata dictionaries

        Raises:
            SessionStorageError: If listing fails
        """
        try:
            sessions = await self._storage.list_sessions()
            if limit and len(sessions) > limit:
                sessions = sessions[:limit]
            logger.debug(f"Listed {len(sessions)} sessions via SessionManager.")
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {e}", exc_info=True)
            raise SessionStorageError(f"Failed to list sessions: {e}")

    async def get_session(self, session_id: str):
        """
        Retrieves a specific chat session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            ChatSession object

        Raises:
            SessionNotFoundError: If the session doesn't exist
            SessionStorageError: If retrieval fails
        """
        try:
            session = await self._storage.get_session(session_id)
            if not session:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            logger.debug(f"Retrieved session '{session_id}' via SessionManager.")
            return session
        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to retrieve session: {e}")

    async def delete_session(self, session_id: str) -> None:
        """
        Deletes a chat session.

        Args:
            session_id: The session ID to delete

        Raises:
            SessionNotFoundError: If the session doesn't exist
            SessionStorageError: If deletion fails
        """
        try:
            success = await self._storage.delete_session(session_id)
            if not success:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            logger.debug(f"Deleted session '{session_id}' via SessionManager.")
        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deleting session '{session_id}': {e}", exc_info=True)
            raise SessionStorageError(f"Failed to delete session: {e}")
'''

# Find the last method's closing and insert before the file ends
# We'll insert right before the final closing of the class
last_method_end = content.rfind('\n        logger.debug(f"Session \'{session.id}\' saved via SessionManager.")')
if last_method_end == -1:
    print("❌ Could not find method insertion point")
    sys.exit(1)

# Move to the end of that line
insert_pos = content.find('\n', last_method_end + 1)

# Insert the new methods
new_content = content[:insert_pos] + new_methods + content[insert_pos:]

# Write back
with open('src/llmcore/sessions/manager.py', 'w') as f:
    f.write(new_content)

print("✅ Added list_sessions, get_session, and delete_session methods to SessionManager")
print("   - list_sessions(): Lists session metadata")
print("   - get_session(): Retrieves a specific session")
print("   - delete_session(): Deletes a session")
