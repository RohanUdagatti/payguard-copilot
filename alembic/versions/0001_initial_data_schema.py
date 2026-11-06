"""Initial PayGuard data schema (merchants, cards, reason_codes, transactions, disputes).

Revision ID: 0001_initial_data_schema
Revises:
Create Date: 2026-11-06 10:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_data_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. merchants
    op.create_table(
        "merchants",
        sa.Column("merchant_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("mcc", sa.String(length=4), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 2. cards
    op.create_table(
        "cards",
        sa.Column("card_token", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("card_brand", sa.String(length=16), nullable=False),
        sa.Column("card_type", sa.String(length=16), nullable=False),
        sa.Column("issuer_country", sa.String(length=2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 3. reason_codes
    op.create_table(
        "reason_codes",
        sa.Column("reason_code", sa.String(length=16), primary_key=True, nullable=False),
        sa.Column("network", sa.String(length=16), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
    )

    # 4. transactions
    op.create_table(
        "transactions",
        sa.Column("transaction_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("card_token", sa.String(length=64), sa.ForeignKey("cards.card_token"), nullable=False),
        sa.Column("merchant_id", sa.String(length=64), sa.ForeignKey("merchants.merchant_id"), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="USD"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("txn_hour", sa.SmallInteger(), nullable=False),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column("mcc", sa.String(length=4), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("is_international", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("is_fraud", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_transactions_card_timestamp", "transactions", ["card_token", "timestamp"])
    op.create_index("idx_transactions_merchant_timestamp", "transactions", ["merchant_id", "timestamp"])
    op.create_index("idx_transactions_timestamp_hour", "transactions", ["timestamp", "txn_hour"])
    op.create_index("idx_transactions_is_fraud", "transactions", ["is_fraud"])

    # 5. disputes
    op.create_table(
        "disputes",
        sa.Column("dispute_id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("transaction_id", sa.String(length=64), sa.ForeignKey("transactions.transaction_id"), nullable=False, unique=True),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="USD"),
        sa.Column("reason_code", sa.String(length=16), sa.ForeignKey("reason_codes.reason_code"), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("evidence_due_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_disputes_transaction_id", "disputes", ["transaction_id"])
    op.create_index("idx_disputes_status_filed_at", "disputes", ["status", "filed_at"])
    op.create_index("idx_disputes_reason_code", "disputes", ["reason_code"])


def downgrade() -> None:
    op.drop_index("idx_disputes_reason_code", table_name="disputes")
    op.drop_index("idx_disputes_status_filed_at", table_name="disputes")
    op.drop_index("idx_disputes_transaction_id", table_name="disputes")
    op.drop_table("disputes")

    op.drop_index("idx_transactions_is_fraud", table_name="transactions")
    op.drop_index("idx_transactions_timestamp_hour", table_name="transactions")
    op.drop_index("idx_transactions_merchant_timestamp", table_name="transactions")
    op.drop_index("idx_transactions_card_timestamp", table_name="transactions")
    op.drop_table("transactions")

    op.drop_table("reason_codes")
    op.drop_table("cards")
    op.drop_table("merchants")

