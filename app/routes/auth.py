"""
Authentication Routes

Handles user authentication, registration, and role management.
Uses JWT tokens for session management.
"""
from typing import Dict, Any, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connections import get_postgres_session
from app.db.models import User
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# === Request/Response Models ===

class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str
    role: Optional[str] = "user"


class UserResponse(BaseModel):
    """User data response."""
    id: str
    email: str
    role: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str


# === Helper Functions ===

async def get_user_by_email(email: str) -> Optional[User]:
    """Fetch user by email from database."""
    async with get_postgres_session() as session:
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: UUID) -> Optional[User]:
    """Fetch user by ID from database."""
    async with get_postgres_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()


# === Authentication Endpoints ===

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister) -> TokenResponse:
    """
    Register a new user.
    
    Creates user account and returns JWT token.
    """
    # Check if user already exists
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create new user
    async with get_postgres_session() as session:
        new_user = User(
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            role="user",  # Public registration always creates "user" role; admin is set via admin endpoint only
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        
        # Generate token
        token = create_access_token(
            data={
                "sub": str(new_user.id),
                "email": new_user.email,
                "role": new_user.role,
            }
        )
        
        logger.info(f"User registered: {new_user.email}")
        
        return TokenResponse(
            access_token=token,
            user=UserResponse(
                id=str(new_user.id),
                email=new_user.email,
                role=new_user.role,
            ),
        )


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """
    Authenticate user and return JWT token.
    
    Uses OAuth2 password flow for compatibility.
    """
    # Find user
    user = await get_user_by_email(form_data.username)
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
        }
    )
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=token,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            role=user.role,
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
) -> UserResponse:
    """Get current authenticated user's information."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        role=current_user["role"],
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Change the current user's password."""
    user = await get_user_by_id(UUID(current_user["id"]))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Verify current password
    if not verify_password(password_data.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    
    # Update password
    async with get_postgres_session() as session:
        result = await session.execute(
            select(User).where(User.id == UUID(current_user["id"]))
        )
        db_user = result.scalar_one()
        db_user.password_hash = hash_password(password_data.new_password)
        await session.commit()
    
    logger.info(f"Password changed for: {current_user['email']}")
    
    return {"message": "Password updated successfully"}


# === Admin Endpoints ===

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: dict = Depends(get_current_admin),
) -> list[UserResponse]:
    """
    List all users (admin only).
    """
    async with get_postgres_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        
        return [
            UserResponse(
                id=str(user.id),
                email=user.email,
                role=user.role,
            )
            for user in users
        ]


@router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: UUID,
    role: str,
    current_user: dict = Depends(get_current_admin),
) -> Dict[str, Any]:
    """
    Update a user's role (admin only).
    """
    if role not in ["admin", "user"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'admin' or 'user'",
        )
    
    async with get_postgres_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        user.role = role
        await session.commit()
        
        logger.info(f"Role updated for {user.email}: {role}")
        
        return {"message": f"User role updated to {role}"}
