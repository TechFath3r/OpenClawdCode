"""Composite decay scoring engine.

Port of memory-lancedb-pro's decay-engine.ts. Computes a composite score
for a memory based on recency (Weibull stretched-exponential), access
frequency, and intrinsic value. The composite is then used to modulate
search similarity scores.

Reference: references/source-algorithms.md § Weibull decay
Source:    CortexReach/memory-lancedb-pro src/decay-engine.ts:17-232
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Constants ---

SECS_PER_DAY = 86_400.0

# --- Config (matches DEFAULT_DECAY_CONFIG) ---

HALF_LIFE_DAYS = 30.0          # τ — base recency half-life
IMPORTANCE_MODULATION = 1.5    # μ — effective_hl = τ * exp(μ * importance)

TIER_BETA: dict[str, float] = {
    "core": 0.8,
    "working": 1.0,
    "peripheral": 1.3,
}

TIER_FLOOR: dict[str, float] = {
    "core": 0.9,
    "working": 0.7,
    "peripheral": 0.5,
}

RECENCY_WEIGHT = 0.4
FREQUENCY_WEIGHT = 0.3
INTRINSIC_WEIGHT = 0.3

SEARCH_BOOST_MIN = 0.3
STALE_THRESHOLD = 0.3


# --- Data ---

@dataclass(frozen=True, slots=True)
class DecayableMemory:
    """Minimal fields needed to compute a decay score."""
    importance: int            # 1..10
    confidence: float          # 0..1
    tier: str                  # core|working|peripheral
    temporal_type: str         # static|dynamic
    access_count: int
    created_at: float          # epoch seconds
    last_accessed_at: float    # epoch seconds


# --- Scoring functions ---

def recency_score(mem: DecayableMemory, now: float) -> float:
    """Weibull stretched-exponential recency decay.

    Returns 1.0 for a brand-new memory, decaying toward 0 over days.
    Dynamic memories decay 3x faster (1/3 base half-life).
    Higher importance extends the effective half-life exponentially.
    """
    last_active = mem.last_accessed_at if mem.access_count > 0 else mem.created_at
    days_since = max(0.0, (now - last_active) / SECS_PER_DAY)

    base_hl = HALF_LIFE_DAYS / 3.0 if mem.temporal_type == "dynamic" else HALF_LIFE_DAYS
    effective_hl = base_hl * math.exp(IMPORTANCE_MODULATION * mem.importance)
    lam = math.log(2) / effective_hl
    beta = TIER_BETA.get(mem.tier, TIER_BETA["working"])

    return math.exp(-lam * (days_since ** beta))


def frequency_score(mem: DecayableMemory, now: float) -> float:
    """Frequency component: reward memories accessed often and recently.

    base = 1 - exp(-accessCount / 5)       — saturates after ~15 accesses
    recentnessBonus = exp(-avgGap / 30)     — penalizes infrequent access
    frequency = base * (0.5 + 0.5 * bonus)  — blend

    Returns 0.0 when access_count == 0 (never recalled).
    """
    if mem.access_count == 0:
        return 0.0

    base = 1.0 - math.exp(-mem.access_count / 5.0)

    # Average gap between creation and last access
    span_days = max(0.0, (mem.last_accessed_at - mem.created_at) / SECS_PER_DAY)
    avg_gap = span_days / mem.access_count if mem.access_count > 0 else 0.0
    recentness_bonus = math.exp(-avg_gap / 30.0)

    return base * (0.5 + 0.5 * recentness_bonus)


def intrinsic_score(mem: DecayableMemory) -> float:
    """Intrinsic value: importance × confidence, normalized to ~[0, 1].

    importance is 1..10, confidence is 0..1 → raw product is 0..10.
    We normalize by dividing by 10 so the composite weights stay balanced.
    """
    return (mem.importance * mem.confidence) / 10.0


def composite_score(mem: DecayableMemory, now: float) -> float:
    """Weighted composite of recency, frequency, and intrinsic value.

    composite = 0.4 * recency + 0.3 * frequency + 0.3 * intrinsic
    """
    r = recency_score(mem, now)
    f = frequency_score(mem, now)
    i = intrinsic_score(mem)
    return RECENCY_WEIGHT * r + FREQUENCY_WEIGHT * f + INTRINSIC_WEIGHT * i


def apply_search_boost(
    search_score: float,
    mem: DecayableMemory,
    now: float,
) -> float:
    """Modulate a vector/hybrid search score by the decay composite.

    multiplier = boostMin + (1 - boostMin) * max(tierFloor, composite)
    result     = search_score * clamp(multiplier, boostMin, 1.0)

    The tier floor prevents core memories from being penalized even if
    their composite is momentarily low (e.g. not accessed recently).
    """
    comp = composite_score(mem, now)
    tier_floor = TIER_FLOOR.get(mem.tier, TIER_FLOOR["working"])
    effective = max(tier_floor, comp)
    multiplier = SEARCH_BOOST_MIN + (1.0 - SEARCH_BOOST_MIN) * effective
    multiplier = min(1.0, max(SEARCH_BOOST_MIN, multiplier))
    return search_score * multiplier


def is_stale(mem: DecayableMemory, now: float) -> bool:
    """True if the composite score has fallen below the stale threshold."""
    return composite_score(mem, now) < STALE_THRESHOLD
