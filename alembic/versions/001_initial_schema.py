"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2026-02-03

Creates the 'neural_nexus' schema and all application tables.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Schema name for Neural Nexus
SCHEMA_NAME = 'neural_nexus'


def upgrade() -> None:
    # Create the neural_nexus schema
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
    
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.String(20), server_default='user'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.CheckConstraint("role IN ('admin', 'user')", name='valid_role'),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_users_email', 'users', ['email'], schema=SCHEMA_NAME)
    
    # Create folders table
    op.create_table(
        'folders',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_folders_user', 'folders', ['user_id'], schema=SCHEMA_NAME)
    
    # Create files table
    op.create_table(
        'files',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('folder_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.folders.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('azure_blob_url', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('node_count', sa.Integer(), server_default='0'),
        sa.Column('relationship_count', sa.Integer(), server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'processing', 'ready_for_review', 'completed', 'failed')", name='valid_status'),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_files_folder', 'files', ['folder_id'], schema=SCHEMA_NAME)
    op.create_index('idx_files_status', 'files', ['status'], schema=SCHEMA_NAME)
    
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id'), nullable=False),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=True),
        sa.Column('target_id', sa.String(255), nullable=True),
        sa.Column('old_value', postgresql.JSONB(), nullable=True),
        sa.Column('new_value', postgresql.JSONB(), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()')),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_audit_user_time', 'audit_logs', ['user_id', 'timestamp'], schema=SCHEMA_NAME)
    op.create_index('idx_audit_action', 'audit_logs', ['action'], schema=SCHEMA_NAME)
    
    # Create chat_history table
    op.create_table(
        'chat_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('citations', postgresql.JSONB(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()')),
        sa.CheckConstraint("role IN ('user', 'assistant', 'web_search')", name='valid_chat_role'),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_chat_session', 'chat_history', ['session_id', 'timestamp'], schema=SCHEMA_NAME)
    op.create_index('idx_chat_user', 'chat_history', ['user_id'], schema=SCHEMA_NAME)
    
    # Create entity_staging table for Review Inbox
    op.create_table(
        'entity_staging',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('file_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('entity_data', postgresql.JSONB(), nullable=False),
        sa.Column('relationship_data', postgresql.JSONB(), nullable=False),
        sa.Column('validation_status', sa.String(20), server_default='pending'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.CheckConstraint("validation_status IN ('pending', 'approved', 'rejected')", name='valid_staging_status'),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_staging_file', 'entity_staging', ['file_id'], schema=SCHEMA_NAME)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('entity_staging', schema=SCHEMA_NAME)
    op.drop_table('chat_history', schema=SCHEMA_NAME)
    op.drop_table('audit_logs', schema=SCHEMA_NAME)
    op.drop_table('files', schema=SCHEMA_NAME)
    op.drop_table('folders', schema=SCHEMA_NAME)
    op.drop_table('users', schema=SCHEMA_NAME)
    
    # Drop the schema
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA_NAME}")
