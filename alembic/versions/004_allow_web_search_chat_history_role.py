"""Allow web_search role in chat_history

Revision ID: 004
Revises: 003
Create Date: 2026-04-02

Updates the chat_history role constraint so persisted web search rows can be
stored alongside user and assistant messages.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


SCHEMA_NAME = 'neural_nexus'
TABLE_NAME = 'chat_history'
LEGACY_CONSTRAINT_NAME = 'valid_chat_role'
CONVENTION_CONSTRAINT_NAME = 'ck_chat_history_valid_chat_role'


def upgrade() -> None:
    op.execute(f'ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP CONSTRAINT IF EXISTS {LEGACY_CONSTRAINT_NAME}')
    op.execute(f'ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP CONSTRAINT IF EXISTS {CONVENTION_CONSTRAINT_NAME}')
    op.create_check_constraint(
        LEGACY_CONSTRAINT_NAME,
        TABLE_NAME,
        "role IN ('user', 'assistant', 'web_search')",
        schema=SCHEMA_NAME,
    )


def downgrade() -> None:
    op.execute(f'ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP CONSTRAINT IF EXISTS {LEGACY_CONSTRAINT_NAME}')
    op.execute(f'ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME} DROP CONSTRAINT IF EXISTS {CONVENTION_CONSTRAINT_NAME}')
    op.create_check_constraint(
        LEGACY_CONSTRAINT_NAME,
        TABLE_NAME,
        "role IN ('user', 'assistant')",
        schema=SCHEMA_NAME,
    )
