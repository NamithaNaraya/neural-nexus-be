"""Add weight_configs table

Revision ID: 003
Revises: 002
Create Date: 2026-03-05

Adds the 'weight_configs' table for dynamic quantitative weight definitions per folder.
Supports formula storage as JSONB and per-folder activation (global toggle).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Schema name for Neural Nexus
SCHEMA_NAME = 'neural_nexus'


def upgrade() -> None:
    op.create_table(
        'weight_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('folder_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.folders.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey(f'{SCHEMA_NAME}.users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('formula', postgresql.JSONB(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        schema=SCHEMA_NAME,
    )
    op.create_index('idx_wc_folder', 'weight_configs', ['folder_id'], schema=SCHEMA_NAME)
    op.create_index('idx_wc_user', 'weight_configs', ['user_id'], schema=SCHEMA_NAME)


def downgrade() -> None:
    op.drop_table('weight_configs', schema=SCHEMA_NAME)
