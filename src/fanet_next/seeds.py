"""Seed ranges that keep training, development and test scenes apart."""

from __future__ import annotations

TRAIN_SEED_BASE = 1_000_000
DEV_SEED_BASE = 10_000_000
TEST_SEED_BASE = 20_000_000


def split_of(seed: int) -> str:
    """'test', 'dev' or 'train' (every seed below the dev range, ad-hoc ones included)."""
    if seed >= TEST_SEED_BASE:
        return "test"
    if seed >= DEV_SEED_BASE:
        return "dev"
    return "train"
