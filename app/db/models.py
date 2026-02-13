"""
SQLAlchemy ORM Models

Defines the PostgreSQL database schema for:
- Users (authentication & roles)
- Folders (topic organization)
- Files (uploaded documents)
- AuditLogs (activity tracking)
- ChatHistory (conversation persistence)

All tables are created in the 'neural_nexus' schema.
"""
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    CheckConstraint,
    Index,
    JSON,
    MetaData,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

# Schema name for all Neural Nexus tables
SCHEMA_NAME = "neural_nexus"

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    metadata = MetaData(naming_convention=convention, schema=SCHEMA_NAME)


class User(Base):
    """
    User account model.
    
    Stores authentication credentials and role information.
    Roles: 'admin' or 'user'
    """
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        default="user",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    folders: Mapped[List["Folder"]] = relationship(
        "Folder",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
    )
    chat_history: Mapped[List["ChatHistory"]] = relationship(
        "ChatHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    encounters: Mapped[List["Encounter"]] = relationship(
        "Encounter",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'user')", name="valid_role"),
    )


class Folder(Base):
    """
    Folder/Topic model.
    
    Organizes files into logical topic groups.
    Each folder belongs to a single user.
    """
    __tablename__ = "folders"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="folders")
    files: Mapped[List["File"]] = relationship(
        "File",
        back_populates="folder",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        Index("idx_folders_user", "user_id"),
    )


class File(Base):
    """
    File model.
    
    Tracks uploaded documents and their processing status.
    Status flow: pending -> processing -> ready_for_review -> completed
    """
    __tablename__ = "files"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    folder_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("folders.id", ondelete="CASCADE"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    file_size: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    azure_blob_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
    )
    node_count: Mapped[int] = mapped_column(
        default=0,
    )
    relationship_count: Mapped[int] = mapped_column(
        default=0,
    )
    progress: Mapped[int] = mapped_column(
        default=0,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )
    
    # Relationships
    folder: Mapped["Folder"] = relationship("Folder", back_populates="files")
    
    __table_args__ = (
        Index("idx_files_folder", "folder_id"),
        Index("idx_files_status", "status"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready_for_review', 'completed', 'failed')",
            name="valid_status",
        ),
    )


class AuditLog(Base):
    """
    Audit log model.
    
    Tracks all user actions for compliance and debugging.
    Stores old/new values as JSON for detailed change tracking.
    """
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    target_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    target_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    old_value: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )
    new_value: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="audit_logs")
    
    __table_args__ = (
        Index("idx_audit_user_time", "user_id", "timestamp"),
        Index("idx_audit_action", "action"),
    )


class ChatHistory(Base):
    """
    Chat history model.
    
    Persists all AI conversations for audit and context.
    Uses session_id to group conversations.
    Implements Sliding Window context (last 5 Q&A pairs).
    """
    __tablename__ = "chat_history"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    citations: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_history")
    
    __table_args__ = (
        Index("idx_chat_session", "session_id", "timestamp"),
        Index("idx_chat_user", "user_id"),
        CheckConstraint("role IN ('user', 'assistant')", name="valid_chat_role"),
    )


# Entity staging table for "Review Inbox" feature
class EntityStaging(Base):
    """
    Entity staging model.
    
    Holds extracted entities pending user approval.
    Part of the 7-phase ingestion pipeline "Quality Gate".
    """
    __tablename__ = "entity_staging"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    relationship_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
    )
    validation_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    __table_args__ = (
        Index("idx_staging_file", "file_id"),
        CheckConstraint(
            "validation_status IN ('pending', 'approved', 'rejected')",
            name="valid_staging_status",
        ),
    )


class Encounter(Base):
    """
    Reasoning Encounter model.
    
    Persists a single reasoning/consultation session.
    Links to input indicators, inferred states, and outcomes.
    """
    __tablename__ = "encounters"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    indicators: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    inferred_states: Mapped[dict] = mapped_column(
        JSON,
        nullable=True,
    )
    recommendations: Mapped[dict] = mapped_column(
        JSON,
        nullable=True,
    )
    safety_status: Mapped[str] = mapped_column(
        String(20),
        default="unknown",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    user: Mapped["User"] = relationship("User", back_populates="encounters")
    outcomes: Mapped[List["Outcome"]] = relationship("Outcome", back_populates="encounter")


class Outcome(Base):
    """
    Encounter Outcome/Feedback model.
    
    Stores patient/user feedback for Step 13.
    Used for long-term effectiveness tracking.
    """
    __tablename__ = "outcomes"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    encounter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("encounters.id", ondelete="CASCADE"),
        nullable=False,
    )
    feedback: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    rating: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    
    encounter: Mapped["Encounter"] = relationship("Encounter", back_populates="outcomes")
