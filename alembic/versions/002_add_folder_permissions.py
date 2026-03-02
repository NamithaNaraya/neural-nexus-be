"""Add folder_permissions table

Revision ID: 002
Revises: 001
Create Date: 2026-03-02

Adds the 'folder_permissions' table for sharing folders between users.
Supports 'read' and 'write' permission levels with UPSERT-friendly unique constraint.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Schema name for Neural Nexus
SCHEMA_NAME = 'neural_nexus'


def upgrade() -> None:
    op.create_table(
        'folder_permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('folder_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.folders.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('permission', sa.String(20), nullable=False, server_default='read'),
        sa.Column('granted_by', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id'), nullable=True),
        sa.Column('granted_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.CheckConstraint("permission IN ('read', 'write', 'admin')", name='valid_folder_permission'),
        sa.UniqueConstraint('folder_id', 'user_id', name='uq_folder_user_permission'),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_fp_folder', 'folder_permissions', ['folder_id'], schema=SCHEMA_NAME)
    op.create_index('idx_fp_user', 'folder_permissions', ['user_id'], schema=SCHEMA_NAME)


def downgrade() -> None:
    op.drop_table('folder_permissions', schema=SCHEMA_NAME)
