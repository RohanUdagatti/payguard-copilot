"""Lightweight input checks for the RAG gateway (full Presidio comes later)."""

from __future__ import annotations

import re

_DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def luhn_checksum_valid(digits: str) -> bool:
    """Return True if ``digits`` pass the Luhn check."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    reverse = digits[::-1]
    for idx, ch in enumerate(reverse):
        n = ord(ch) - 48
        if idx % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def contains_luhn_pan(text: str) -> bool:
    """Return True if ``text`` contains a Luhn-valid 13–19 digit card number."""
    for match in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if luhn_checksum_valid(digits):
            return True
    return False
