"""SQLAlchemy data models for PayGuard Phase 1 schema."""

from __future__ import annotations

import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for PayGuard data models."""

    pass


class Merchant(Base):
    """Merchant entity with industry classification and risk tier."""

    __tablename__ = "merchants"

    merchant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mcc: Mapped[str] = mapped_column(String(4), nullable=False)
    risk_tier: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # LOW, MEDIUM, HIGH
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    transactions: Mapped[list[Transaction]] = relationship(back_populates="merchant")


class Card(Base):
    """Tokenized card account entity. Strictly no raw PAN stored."""

    __tablename__ = "cards"

    card_token: Mapped[str] = mapped_column(String(64), primary_key=True)  # tok_...
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    card_brand: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # VISA, MASTERCARD, AMEX, DISCOVER
    card_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # CREDIT, DEBIT, PREPAID
    issuer_country: Mapped[str] = mapped_column(String(2), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    transactions: Mapped[list[Transaction]] = relationship(back_populates="card")


class ReasonCode(Base):
    """Chargeback reason codes (Visa, Mastercard) seeded for disputes."""

    __tablename__ = "reason_codes"

    reason_code: Mapped[str] = mapped_column(
        String(16), primary_key=True
    )  # e.g. 10.4, 13.1, 4837
    network: Mapped[str] = mapped_column(String(16), nullable=False)  # VISA, MASTERCARD
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # FRAUD, AUTHORIZATION, PROCESSING_ERROR, CUSTOMER_DISPUTE

    disputes: Mapped[list[Dispute]] = relationship(back_populates="reason")


class Transaction(Base):
    """Payment transaction entity containing features for downstream fraud modeling."""

    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # txn_...
    card_token: Mapped[str] = mapped_column(
        String(64), ForeignKey("cards.card_token"), nullable=False
    )
    merchant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("merchants.merchant_id"), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    txn_hour: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # 0-23
    channel: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # POS, ECOMMERCE, ATM, MOTO
    mcc: Mapped[str] = mapped_column(String(4), nullable=False)
    risk_tier: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # LOW, MEDIUM, HIGH
    is_international: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # APPROVED, DECLINED, REFUNDED
    is_fraud: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )  # Ground truth label
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    card: Mapped[Card] = relationship(back_populates="transactions")
    merchant: Mapped[Merchant] = relationship(back_populates="transactions")
    dispute: Mapped[Dispute | None] = relationship(
        back_populates="transaction", uselist=False
    )

    __table_args__ = (
        Index("idx_transactions_card_timestamp", "card_token", "timestamp"),
        Index("idx_transactions_merchant_timestamp", "merchant_id", "timestamp"),
        Index("idx_transactions_timestamp_hour", "timestamp", "txn_hour"),
        Index("idx_transactions_is_fraud", "is_fraud"),
    )


class Dispute(Base):
    """Chargeback / dispute entity referencing a transaction and reason code."""

    __tablename__ = "disputes"

    dispute_id: Mapped[str] = mapped_column(String(64), primary_key=True)  # dsp_...
    transaction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("transactions.transaction_id"),
        nullable=False,
        unique=True,
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    reason_code: Mapped[str] = mapped_column(
        String(16), ForeignKey("reason_codes.reason_code"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(24), nullable=False
    )  # OPEN, UNDER_REVIEW, WON, LOST
    evidence_due_date: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    filed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    resolved_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    transaction: Mapped[Transaction] = relationship(back_populates="dispute")
    reason: Mapped[ReasonCode] = relationship(back_populates="disputes")

    __table_args__ = (
        Index("idx_disputes_transaction_id", "transaction_id"),
        Index("idx_disputes_status_filed_at", "status", "filed_at"),
        Index("idx_disputes_reason_code", "reason_code"),
    )
