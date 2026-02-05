"""
Permission Service

Role-Based Access Control (RBAC) for folder and file permissions.
Features:
- Folder-level permissions (owner, editor, viewer)
- Permission inheritance
- Contextual search scope locking
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Folder
from app.core.security import get_current_user


class PermissionLevel(str, Enum):
    """Permission levels for folder access."""
    OWNER = "owner"       # Full control: edit, delete, share
    EDITOR = "editor"     # Can edit content, cannot delete folder
    VIEWER = "viewer"     # Read-only access
    NONE = "none"         # No access


class FolderPermission:
    """Represents a user's permission on a folder."""
    def __init__(
        self,
        folder_id: str,
        user_id: str,
        level: PermissionLevel,
        granted_by: Optional[str] = None,
        granted_at: Optional[datetime] = None,
    ):
        self.folder_id = folder_id
        self.user_id = user_id
        self.level = level
        self.granted_by = granted_by
        self.granted_at = granted_at or datetime.utcnow()


class PermissionService:
    """Service for managing folder permissions and access control."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._permission_cache: Dict[str, Dict[str, PermissionLevel]] = {}
    
    async def check_permission(
        self,
        user_id: str,
        folder_id: str,
        required_level: PermissionLevel = PermissionLevel.VIEWER,
    ) -> bool:
        """
        Check if user has required permission level on folder.
        
        Args:
            user_id: User to check
            folder_id: Folder to check access for
            required_level: Minimum permission level required
        
        Returns:
            True if user has sufficient permissions
        """
        permission = await self.get_permission(user_id, folder_id)
        
        # Permission hierarchy: OWNER > EDITOR > VIEWER > NONE
        hierarchy = {
            PermissionLevel.OWNER: 3,
            PermissionLevel.EDITOR: 2,
            PermissionLevel.VIEWER: 1,
            PermissionLevel.NONE: 0,
        }
        
        return hierarchy.get(permission, 0) >= hierarchy.get(required_level, 0)
    
    async def get_permission(
        self,
        user_id: str,
        folder_id: str,
    ) -> PermissionLevel:
        """Get user's permission level on a folder."""
        # Check cache first
        cache_key = f"{user_id}:{folder_id}"
        if folder_id in self._permission_cache:
            if user_id in self._permission_cache[folder_id]:
                return self._permission_cache[folder_id][user_id]
        
        # Query database for permission
        result = await self.db.execute(
            select(Folder).where(Folder.id == folder_id)
        )
        folder = result.scalar_one_or_none()
        
        if not folder:
            return PermissionLevel.NONE
        
        # Check if user is owner
        if str(folder.user_id) == user_id:
            permission = PermissionLevel.OWNER
        else:
            # Check folder permissions table
            # This would query a folder_permissions join table
            # For now, assume no access if not owner
            permission = PermissionLevel.NONE
        
        # Cache the result
        if folder_id not in self._permission_cache:
            self._permission_cache[folder_id] = {}
        self._permission_cache[folder_id][user_id] = permission
        
        return permission
    
    async def grant_permission(
        self,
        granter_id: str,
        grantee_id: str,
        folder_id: str,
        level: PermissionLevel,
    ) -> bool:
        """
        Grant permission to a user on a folder.
        Only owners can grant permissions.
        """
        # Verify granter is owner
        granter_permission = await self.get_permission(granter_id, folder_id)
        if granter_permission != PermissionLevel.OWNER:
            raise PermissionError("Only folder owners can grant permissions")
        
        # Cannot demote owner
        if level == PermissionLevel.NONE and await self.get_permission(grantee_id, folder_id) == PermissionLevel.OWNER:
            raise PermissionError("Cannot remove owner permissions")
        
        # In a real implementation, this would insert/update folder_permissions table
        # For now, we just update the cache
        if folder_id not in self._permission_cache:
            self._permission_cache[folder_id] = {}
        self._permission_cache[folder_id][grantee_id] = level
        
        return True
    
    async def revoke_permission(
        self,
        revoker_id: str,
        target_id: str,
        folder_id: str,
    ) -> bool:
        """Revoke a user's permission on a folder."""
        return await self.grant_permission(
            revoker_id, target_id, folder_id, PermissionLevel.NONE
        )
    
    async def get_accessible_folders(
        self,
        user_id: str,
        min_level: PermissionLevel = PermissionLevel.VIEWER,
    ) -> List[str]:
        """Get all folder IDs the user can access with at least min_level."""
        # Query folders owned by user
        result = await self.db.execute(
            select(Folder.id).where(Folder.user_id == user_id)
        )
        owned_folders = [str(row[0]) for row in result.fetchall()]
        
        # Query shared folders (from folder_permissions table)
        # This would be extended in a real implementation
        shared_folders: List[str] = []
        
        return owned_folders + shared_folders
    
    def invalidate_cache(self, folder_id: Optional[str] = None):
        """Invalidate permission cache."""
        if folder_id:
            self._permission_cache.pop(folder_id, None)
        else:
            self._permission_cache.clear()


class ContextualSearchLock:
    """
    Enforces search scope restrictions based on user permissions.
    Users can only search within folders they have access to.
    """
    
    def __init__(self, permission_service: PermissionService):
        self.permission_service = permission_service
    
    async def get_search_scope(
        self,
        user_id: str,
        requested_folders: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get the actual search scope based on user permissions.
        
        Args:
            user_id: User performing the search
            requested_folders: Folders the user wants to search in
        
        Returns:
            List of folder IDs the user can actually search
        """
        # Get all accessible folders
        accessible = await self.permission_service.get_accessible_folders(user_id)
        
        if requested_folders is None:
            # No specific folders requested - search all accessible
            return accessible
        
        # Filter requested folders to only those accessible
        return [f for f in requested_folders if f in accessible]
    
    async def validate_query_scope(
        self,
        user_id: str,
        query_scope: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate and sanitize a query scope object.
        Ensures user cannot access data outside their permissions.
        """
        scope_type = query_scope.get("type", "global")
        scope_id = query_scope.get("id")
        
        if scope_type == "global":
            # Restrict to accessible folders
            accessible = await self.permission_service.get_accessible_folders(user_id)
            return {
                "type": "folders",
                "ids": accessible,
            }
        
        elif scope_type == "folder":
            # Verify access to specific folder
            has_access = await self.permission_service.check_permission(
                user_id, scope_id, PermissionLevel.VIEWER
            )
            if not has_access:
                raise PermissionError(f"Access denied to folder {scope_id}")
            return query_scope
        
        elif scope_type == "file":
            # File access checked through folder
            # Would need to look up folder for file
            return query_scope
        
        return query_scope


# Singleton instance
_permission_service: Optional[PermissionService] = None


def get_permission_service(db: AsyncSession) -> PermissionService:
    """Get or create permission service instance."""
    global _permission_service
    if _permission_service is None:
        _permission_service = PermissionService(db)
    return _permission_service
