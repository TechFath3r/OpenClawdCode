"""Tests for the decay engine — verified against hand-calculated reference values.

Reference formulas: references/source-algorithms.md § Weibull decay
"""

import math
import pytest

from openclawd.decay import (
    DecayableMemory,
    SECS_PER_DAY,
    recency_score,
    frequency_score,
    intrinsic_score,
    composite_score,
    apply_search_boost,
    is_stale,
)


def _mem(
    importance=5, confidence=1.0, tier="working", temporal_type="static",
    access_count=0, created_at=0.0, last_accessed_at=0.0,
) -> DecayableMemory:
    return DecayableMemory(
        importance=importance, confidence=confidence, tier=tier,
        temporal_type=temporal_type, access_count=access_count,
        created_at=created_at, last_accessed_at=last_accessed_at,
    )


# --- Recency ---

class TestRecency:
    def test_brand_new_memory_is_one(self):
        """A memory created just now should have recency ≈ 1.0."""
        now = 1000.0
        m = _mem(created_at=now)
        assert recency_score(m, now) == pytest.approx(1.0, abs=1e-9)

    def test_working_tier_imp5_10d(self):
        """Working tier, importance=5, 10 days old.
        effective_hl = 30 * exp(1.5*5) = 54241.27
        λ = ln2 / 54241.27 = 1.2779e-5
        recency = exp(-λ * 10^1.0) ≈ 0.99987
        """
        now = 10 * SECS_PER_DAY
        m = _mem(importance=5, created_at=0.0)
        assert recency_score(m, now) == pytest.approx(0.9998722185, rel=1e-6)

    def test_dynamic_decays_3x_faster(self):
        """Dynamic temporal type uses halfLife/3."""
        now = 10 * SECS_PER_DAY
        static = _mem(importance=5, temporal_type="static", created_at=0.0)
        dynamic = _mem(importance=5, temporal_type="dynamic", created_at=0.0)
        rs = recency_score(static, now)
        rd = recency_score(dynamic, now)
        assert rd < rs  # dynamic decays faster
        assert rd == pytest.approx(0.9996167046, rel=1e-6)

    def test_peripheral_low_imp_100d(self):
        """Peripheral tier, importance=1, 100 days old — should be very low.
        effective_hl = 30 * exp(1.5) ≈ 134.45
        β = 1.3 (peripheral)
        recency = exp(-λ * 100^1.3) ≈ 0.1284
        """
        now = 100 * SECS_PER_DAY
        m = _mem(importance=1, tier="peripheral", created_at=0.0)
        assert recency_score(m, now) == pytest.approx(0.1284260205, rel=1e-4)

    def test_core_decays_slowest(self):
        """Core β < working β < peripheral β → core recency highest at same age."""
        now = 50 * SECS_PER_DAY
        base = dict(importance=3, created_at=0.0)
        core = recency_score(_mem(**base, tier="core"), now)
        working = recency_score(_mem(**base, tier="working"), now)
        peripheral = recency_score(_mem(**base, tier="peripheral"), now)
        assert core > working > peripheral

    def test_last_accessed_used_when_accessed(self):
        """If access_count > 0, age is from last_accessed_at, not created_at."""
        now = 30 * SECS_PER_DAY
        m_old = _mem(access_count=3, created_at=0.0, last_accessed_at=25 * SECS_PER_DAY)
        m_never = _mem(access_count=0, created_at=0.0)
        assert recency_score(m_old, now) > recency_score(m_never, now)

    def test_higher_importance_decays_slower(self):
        now = 90 * SECS_PER_DAY
        low = recency_score(_mem(importance=1, created_at=0.0), now)
        high = recency_score(_mem(importance=9, created_at=0.0), now)
        assert high > low


# --- Frequency ---

class TestFrequency:
    def test_never_accessed_is_zero(self):
        assert frequency_score(_mem(access_count=0), 0.0) == 0.0

    def test_5_accesses_5d_gap(self):
        """access_count=5, created 30d ago, last access 25d ago → avg gap 5d.
        base = 1 - exp(-5/5) = 1 - exp(-1) ≈ 0.6321
        recentness = exp(-5/30) ≈ 0.8465
        freq = 0.6321 * (0.5 + 0.5*0.8465) ≈ 0.5836
        """
        created = 0.0
        last = 25 * SECS_PER_DAY
        now = 30 * SECS_PER_DAY
        m = _mem(access_count=5, created_at=created, last_accessed_at=last)
        assert frequency_score(m, now) == pytest.approx(0.5836, rel=1e-3)

    def test_more_accesses_higher_score(self):
        now = 100 * SECS_PER_DAY
        few = _mem(access_count=2, created_at=0.0, last_accessed_at=50 * SECS_PER_DAY)
        many = _mem(access_count=20, created_at=0.0, last_accessed_at=50 * SECS_PER_DAY)
        assert frequency_score(many, now) > frequency_score(few, now)


# --- Intrinsic ---

class TestIntrinsic:
    def test_importance_times_confidence(self):
        m = _mem(importance=7, confidence=0.8)
        assert intrinsic_score(m) == pytest.approx(0.56, abs=1e-9)

    def test_max_intrinsic(self):
        m = _mem(importance=10, confidence=1.0)
        assert intrinsic_score(m) == pytest.approx(1.0, abs=1e-9)

    def test_zero_confidence(self):
        m = _mem(importance=10, confidence=0.0)
        assert intrinsic_score(m) == pytest.approx(0.0, abs=1e-9)


# --- Composite ---

class TestComposite:
    def test_fresh_no_access(self):
        """Fresh memory, never accessed: recency≈1, frequency=0, intrinsic=imp*conf/10."""
        now = 0.0
        m = _mem(importance=5, confidence=1.0, created_at=0.0)
        comp = composite_score(m, now)
        # 0.4*1.0 + 0.3*0.0 + 0.3*0.5 = 0.55
        assert comp == pytest.approx(0.55, abs=1e-6)

    def test_weights_sum_to_one(self):
        from openclawd.decay import RECENCY_WEIGHT, FREQUENCY_WEIGHT, INTRINSIC_WEIGHT
        assert RECENCY_WEIGHT + FREQUENCY_WEIGHT + INTRINSIC_WEIGHT == pytest.approx(1.0)


# --- Search boost ---

class TestSearchBoost:
    def test_core_high_composite(self):
        """Core tier memory with high composite — score barely reduced.
        comp=0.95, tierFloor=0.9, eff=0.95
        mult = 0.3 + 0.7*0.95 = 0.965
        boosted = 0.85 * 0.965 = 0.82025
        """
        now = 0.0
        # Build a mem that has high composite (fresh, high importance)
        m = _mem(importance=10, confidence=1.0, tier="core", created_at=now)
        boosted = apply_search_boost(0.85, m, now)
        # composite = 0.4*1.0 + 0.3*0.0 + 0.3*1.0 = 0.7
        # but tierFloor=0.9 wins → eff=0.9 → mult=0.3+0.7*0.9=0.93
        assert boosted == pytest.approx(0.85 * 0.93, rel=1e-4)

    def test_peripheral_stale_gets_floor(self):
        """Very stale peripheral memory — boost still ≥ SEARCH_BOOST_MIN * score."""
        from openclawd.decay import SEARCH_BOOST_MIN
        now = 500 * SECS_PER_DAY
        m = _mem(importance=1, confidence=0.5, tier="peripheral", created_at=0.0)
        boosted = apply_search_boost(0.9, m, now)
        assert boosted >= 0.9 * SEARCH_BOOST_MIN

    def test_boost_never_exceeds_original(self):
        now = 0.0
        m = _mem(importance=10, confidence=1.0, tier="core", created_at=now)
        assert apply_search_boost(0.5, m, now) <= 0.5


# --- Staleness ---

class TestStaleness:
    def test_fresh_is_not_stale(self):
        m = _mem(importance=5, confidence=1.0, created_at=0.0)
        assert not is_stale(m, 0.0)

    def test_very_old_low_imp_is_stale(self):
        m = _mem(importance=1, confidence=0.3, tier="peripheral", created_at=0.0)
        assert is_stale(m, 365 * SECS_PER_DAY)
