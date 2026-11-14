import random
from bisect import bisect_left, bisect_right

import pytest

from streamz import Stream
from streamz.bandjoin import (LEFT, RIGHT, BandJoinIndex, Match, band_join,
                              sample_pairs)


def brute_pairs(left, right, tol):
    return [(lid, rid) for lid, lp in left.items()
            for rid, rp in right.items() if abs(lp - rp) <= tol]


def brute_count(left, right, tol):
    return len(brute_pairs(left, right, tol))


def brute_count_sorted(left, right, tol):
    rs = sorted(right.values())
    return sum(bisect_right(rs, p + tol) - bisect_left(rs, p - tol)
               for p in left.values())


def test_basic_counts():
    idx = BandJoinIndex(tolerance=0.5)
    idx.set('left', 'a', 10.0)
    idx.set('left', 'b', 11.0)
    assert idx.total == 0
    idx.set('right', 'x', 10.2)   # matches a only
    assert idx.total == 1
    idx.set('right', 'y', 10.9)   # matches b only
    assert idx.total == 2
    idx.set('right', 'x', 10.5)   # re-price: now matches both a and b
    assert idx.total == 3
    idx.delete('right', 'x')
    assert idx.total == 1
    idx.delete('left', 'b')
    assert idx.total == 0
    assert idx.state('left') == {'a': 10.0}
    assert idx.state('right') == {'y': 10.9}
    assert idx.row_count(LEFT) == 1
    assert idx.row_count(RIGHT) == 1


def test_exact_match_zero_tolerance():
    idx = BandJoinIndex(tolerance=0)
    idx.set('left', 'a', 5)
    idx.set('left', 'b', 6)
    idx.set('right', 'x', 5)
    assert idx.total == 1
    idx.set('right', 'y', 5.0)
    assert idx.total == 2
    idx.set('right', 'z', 5 + 1e-12)
    assert idx.total == 2


def test_duplicate_prices_kept():
    idx = BandJoinIndex(tolerance=0.25)
    for i in range(4):
        idx.set('left', 'l%d' % i, 1.0)
    for i in range(3):
        idx.set('right', 'r%d' % i, 1.1)
    assert idx.total == 4 * 3
    idx.set('right', 'extra', 5.0)
    assert idx.total == 12
    # deleting one duplicate only removes its own pairs
    idx.delete('left', 'l0')
    assert idx.total == 9
    # every distinct record pair must be reachable by sampling
    rng = random.Random(1)
    seen = set()
    for _ in range(2000):
        m = idx.sample_one(rng)
        seen.add((m.left_id, m.right_id))
    assert seen == {(lid, rid) for lid in ('l1', 'l2', 'l3')
                    for rid in ('r0', 'r1', 'r2')}


def test_repr_and_errors():
    idx = BandJoinIndex(tolerance=1)
    assert 'pairs=0' in repr(idx)
    with pytest.raises(ValueError):
        BandJoinIndex(tolerance=-1)
    with pytest.raises(KeyError):
        idx.delete('left', 'missing')
    with pytest.raises(ValueError):
        idx.set('left', 'a', None)
    with pytest.raises(ValueError):
        idx.set('middle', 'a', 1.0)
    with pytest.raises(ValueError):
        idx.sample_one()


def random_ops(seed, n_ops, max_rows, price_choices):
    rng = random.Random(seed)
    idx = BandJoinIndex(tolerance=1.5, seed=seed)
    left, right = {}, {}
    for _ in range(n_ops):
        side = rng.choice([LEFT, RIGHT])
        rows = left if side == LEFT else right
        name = 'left' if side == LEFT else 'right'
        if not rows or (rng.random() < 0.55 and len(rows) < max_rows):
            rid = '%s%d' % (name, rng.randrange(10 ** 6))
            price = rng.choice(price_choices)
            idx.set(side, rid, price)
            rows[rid] = price
        elif rng.random() < 0.5:
            rid = rng.choice(list(rows))
            price = rng.choice(price_choices)
            idx.set(side, rid, price)   # re-price existing id
            rows[rid] = price
        else:
            rid = rng.choice(list(rows))
            idx.delete(side, rid)
            del rows[rid]
        yield idx, left, right


def test_against_brute_force_small():
    # tiny tables, heavy duplicates -> checked after every single op
    prices = [0, 1, 2, 3, 0.5, 1.5, -1, 2.0]
    for idx, left, right in random_ops(seed=3, n_ops=1500, max_rows=25,
                                       price_choices=prices):
        assert idx.total == brute_count(left, right, 1.5)
    assert idx.state(LEFT) == left
    assert idx.state(RIGHT) == right


def test_against_brute_force_larger():
    rng = random.Random(11)
    prices = [rng.uniform(-50, 50) for _ in range(500)]
    n = 0
    for idx, left, right in random_ops(seed=7, n_ops=2500, max_rows=800,
                                       price_choices=prices):
        n += 1
        if n % 50 == 0:
            assert idx.total == brute_count_sorted(left, right, 1.5)
    assert idx.total == brute_count_sorted(left, right, 1.5)
    assert idx.state(LEFT) == left
    assert idx.state(RIGHT) == right


def test_sampling_validity_random_states():
    rng = random.Random(5)
    prices = [rng.uniform(-5, 5) for _ in range(60)]
    for idx, left, right in random_ops(seed=9, n_ops=400, max_rows=60,
                                       price_choices=prices):
        if idx.total == 0:
            continue
        for _ in range(5):
            m = idx.sample_one(rng)
            assert left[m.left_id] == m.left_price
            assert right[m.right_id] == m.right_price
            assert abs(m.left_price - m.right_price) <= 1.5


def test_sampling_is_pair_uniform():
    # degrees: a -> 2, b -> 1, so P(left=a) must be 2/3 and each of the
    # three pairs exactly 1/3
    idx = BandJoinIndex(tolerance=0.5, seed=1)
    idx.set('left', 'a', 0.0)
    idx.set('left', 'b', 0.4)
    idx.set('right', 'x', 0.1)
    idx.set('right', 'y', -0.2)
    idx.set('right', 'z', 5.0)
    assert idx.total == 3

    rng = random.Random(42)
    n = 30000
    counts = {}
    left_a = 0
    for m in idx.sample(n, rng):
        counts[(m.left_id, m.right_id)] = counts.get((m.left_id, m.right_id), 0) + 1
        left_a += m.left_id == 'a'
    assert set(counts) == {('a', 'x'), ('a', 'y'), ('b', 'x')}
    expect = n / 3.0
    chi2 = sum((c - expect) ** 2 / expect for c in counts.values())
    # chi2 with df=2, 99.9% critical value is 13.816
    assert chi2 < 13.816
    assert abs(left_a / n - 2 / 3) < 0.02


def test_sampling_empty_index():
    idx = BandJoinIndex(tolerance=1)
    with pytest.raises(ValueError):
        idx.sample_one()
    idx.set('left', 'a', 1.0)
    with pytest.raises(ValueError):
        idx.sample(3)


def test_streamz_flow():
    left, right = Stream(), Stream()
    join = left.band_join(right, tolerance=0.5)
    totals = join.sink_to_list()

    left.emit(('set', 'a', 10.0))
    right.emit(('set', 'x', 10.2))
    right.emit(('set', 'y', 20.0))
    left.emit(('set', 'b', 10.1))          # matches x too
    left.emit(('set', 'a', 30.0))          # re-price a away
    right.emit(('del', 'y'))
    left.emit(('del', 'a'))
    assert totals == [0, 1, 1, 2, 1, 1, 1]
    assert join.total == 1
    assert join.state('left') == {'b': 10.1}
    assert join.state('right') == {'x': 10.2}
    assert isinstance(join.index, BandJoinIndex)


def test_streamz_event_formats():
    left, right = Stream(), Stream()
    join = band_join(left, right, tolerance=1.0)
    totals = join.sink_to_list()

    left.emit({'op': 'upsert', 'id': 'a', 'price': 1.0})
    left.emit({'op': 'set', 'rid': 'b', 'price': 2.0})
    right.emit(('insert', 'x', 1.5))
    right.emit({'op': 'delete', 'row_id': 'x'})
    # explicit side overrides the upstream the event arrived on
    left.emit(('right', 'set', 'y', 2.5))
    right.emit(('left', 'del', 'b'))
    assert totals == [0, 0, 2, 0, 1, 0]
    assert join.state('right') == {'y': 2.5}
    assert join.state('left') == {'a': 1.0}


def test_streamz_bad_events():
    left, right = Stream(), Stream()
    join = left.band_join(right, tolerance=1.0)  # noqa: F841 (keep a reference)
    with pytest.raises(ValueError):
        left.emit(('set', 'a'))            # missing price
    with pytest.raises(ValueError):
        left.emit(('frobnicate', 'a', 1.0))
    with pytest.raises(ValueError):
        left.emit(('a',))
    with pytest.raises(ValueError):
        left.emit({'op': 'set', 'price': 1.0})   # no id
    with pytest.raises(ValueError):
        band_join(left, left, tolerance=1.0)


def test_sample_pairs_operator():
    left, right = Stream(), Stream()
    join = left.band_join(right, tolerance=0.5)
    trigger = Stream()
    batches = trigger.sample_pairs(join, seed=8).sink_to_list()

    left.emit(('set', 'a', 10.0))
    right.emit(('set', 'x', 10.2))
    right.emit(('set', 'y', 10.4))

    trigger.emit(5)          # five independent draws
    trigger.emit(None)       # one draw
    trigger.emit(0)          # empty batch
    assert [len(b) for b in batches] == [5, 1, 0]
    for m in batches[0] + batches[1]:
        assert isinstance(m, Match)
        assert m.left_id == 'a'
        assert m.right_id in ('x', 'y')
        assert abs(m.left_price - m.right_price) <= 0.5

    right.emit(('del', 'x'))
    right.emit(('del', 'y'))
    trigger.emit(4)          # nothing matches any more
    assert batches[-1] == []


def test_sample_pairs_from_bare_index():
    idx = BandJoinIndex(tolerance=1.0)
    idx.set('left', 'a', 0.0)
    idx.set('right', 'x', 0.5)
    trigger = Stream()
    batches = sample_pairs(trigger, idx, seed=2).sink_to_list()
    trigger.emit(3)
    assert len(batches[0]) == 3


def test_join_sample_method():
    left, right = Stream(), Stream()
    join = left.band_join(right, tolerance=2.0)
    left.emit(('set', 'a', 1.0))
    right.emit(('set', 'x', 1.5))
    rng = random.Random(0)
    pairs = join.sample(10, rng)
    assert len(pairs) == 10
    assert all(m.left_id == 'a' and m.right_id == 'x' for m in pairs)


def test_massive_hit_change_stays_log():
    # one change matching ~every row must not enumerate the hits
    import time
    n = 20000
    idx = BandJoinIndex(tolerance=100.0, seed=1)
    rng = random.Random(4)
    for i in range(n):
        idx.set('left', i, rng.uniform(0, 10))
    start = time.perf_counter()
    for j in range(20):
        idx.set('right', j, 5.0)       # each insert matches all n left rows
        idx.delete('right', j)
    elapsed = time.perf_counter() - start
    assert idx.total == 0
    idx.set('right', 'hit', 5.0)
    assert idx.total == n
    # 40 updates each hitting 20000 rows: lazy range-add keeps this at
    # microseconds per update; a scan would take ~1s
    assert elapsed < 1.0


@pytest.mark.slow
def test_log_scaling():
    import time
    rng = random.Random(6)
    idx = BandJoinIndex(tolerance=2.0, seed=1)
    timings = {}
    rid = 0
    for target in (2 ** 12, 2 ** 15, 2 ** 18):
        while idx.row_count(LEFT) < target:
            idx.set('left', rid, rng.uniform(0, 100))
            idx.set('right', rid, rng.uniform(0, 100))
            rid += 1
        reps = 3
        best = None
        for _ in range(reps):
            start = time.perf_counter()
            for j in range(2000):
                idx.set('left', 'u%d' % j, rng.uniform(0, 100))
                idx.delete('left', 'u%d' % j)
                idx.sample_one(rng)
            t = (time.perf_counter() - start) / 2000
            best = t if best is None else min(best, t)
        timings[target] = best
    # expected cost grows like log n: ratio should stay near 18/12 = 1.5
    assert timings[2 ** 18] / timings[2 ** 12] < 4
