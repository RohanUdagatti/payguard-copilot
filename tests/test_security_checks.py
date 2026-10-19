"""Tests for Luhn PAN rejection helpers."""

from payguard.rag.security_checks import contains_luhn_pan, luhn_checksum_valid


def test_luhn_valid_known_pan() -> None:
    """Visa test PAN 4111111111111111 is Luhn-valid."""
    assert luhn_checksum_valid("4111111111111111")
    assert contains_luhn_pan("Please refund card 4111-1111-1111-1111 now")


def test_luhn_invalid_number() -> None:
    """A 16-digit number that fails Luhn is not treated as PAN."""
    assert not luhn_checksum_valid("4111111111111112")
    assert not contains_luhn_pan("reference 4111111111111112")
