"""Incremental band join with pair counting and uniform sampling.

This module maintains the join of two tables (a *left* and a *right*
channel of quotes) under a band condition ``|left_price - right_price| <=
tolerance`` while rows are inserted, deleted and re-priced by row id.

It provides three things without ever materializing the join:

1. ``BandJoinIndex`` keeps the current state of both sides keyed by row id.
2. The current number of matching pairs is available in ``O(1)`` via
   ``BandJoinIndex.total`` and is re-emitted by the ``band_join`` operator
   after every change.
3. ``BandJoinIndex.sample`` / the ``sample_pairs`` operator draw pairs
   uniformly at random, independently and with replacement, so that every
   matching pair is equally likely.

Algorithm
---------
Each side is stored in a randomized treap keyed by ``(price, seq)`` where
``seq`` is a per-side sequence number, so records sharing the same price
are kept as distinct rows.  Nodes carry subtree sizes (order statistics)
and, on the left tree, a weight equal to the number of right rows
currently matching that left row, plus the subtree sum of weights.

* left insert/delete/re-price: one weight recomputation (a range count on
  the right tree) plus one treap insert/erase -- ``O(log n)`` expected.
* right insert/delete/re-price: a lazy range-add of ``+1``/``-1`` over the
  contiguous key interval of left rows whose window covers the price --
  ``O(log n)`` expected no matter how many rows are hit.
* the pair total is the weight sum at the root of the left tree.
* sampling draws a slot in ``[0, total)``, descends the left tree by
  weight to find the row owning that slot, then selects the residual-th
  matching right row by rank -- ``O(log n)`` expected per pair.

Nothing is expanded into a pair list and no rescan is deferred to sampling
time; all maintenance happens eagerly inside each update.
"""
import random
from collections import namedtuple

from .core import Stream

__all__ = ['LEFT', 'RIGHT', 'Match', 'BandJoinIndex', 'band_join',
           'sample_pairs']

LEFT = 0
RIGHT = 1

_INF = float('inf')

#: A sampled pair of matching rows.
Match = namedtuple('Match', ['left_id', 'left_price', 'right_id', 'right_price'])


# ---------------------------------------------------------------------------
# Randomized treap with subtree counts, subtree weight sums and lazy
# range-add on weights.  Keys are ``(price, seq)`` tuples.
# ---------------------------------------------------------------------------

class _Node(object):
    __slots__ = ('key', 'prio', 'left', 'right', 'cnt', 'w', 'wsum', 'add', 'rid')

    def __init__(self, key, prio, w, rid):
        self.key = key        # (price, seq)
        self.prio = prio      # random treap priority
        self.left = None
        self.right = None
        self.cnt = 1          # subtree size
        self.w = w            # own weight (match degree / 1)
        self.wsum = w         # subtree weight sum
        self.add = 0          # pending lazy weight increment for the subtree
        self.rid = rid        # user row id


def _cnt(n):
    return n.cnt if n is not None else 0


def _wsum(n):
    return n.wsum if n is not None else 0


def _pull(n):
    n.cnt = 1 + _cnt(n.left) + _cnt(n.right)
    n.wsum = n.w + _wsum(n.left) + _wsum(n.right)


def _apply(n, delta):
    if n is not None:
        n.w += delta
        n.wsum += delta * n.cnt
        n.add += delta


def _push(n):
    add = n.add
    if add:
        _apply(n.left, add)
        _apply(n.right, add)
        n.add = 0


def _split(root, key):
    """Split into (keys < key, keys >= key)."""
    if root is None:
        return None, None
    _push(root)
    if key <= root.key:
        left, right = _split(root.left, key)
        root.left = right
        _pull(root)
        return left, root
    else:
        left, right = _split(root.right, key)
        root.right = left
        _pull(root)
        return root, right


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    # Push before descending so a node never ends up underneath a pending
    # lazy tag it was not part of (its weight already reflects the current
    # state, the tag's subtree must stay exactly the tagged key interval).
    _push(a)
    _push(b)
    if a.prio < b.prio:
        a.right = _merge(a.right, b)
        _pull(a)
        return a
    else:
        b.left = _merge(a, b.left)
        _pull(b)
        return b


def _insert(root, node):
    left, right = _split(root, node.key)
    return _merge(_merge(left, node), right)


def _erase(root, key):
    if root is None:
        return None
    _push(root)
    if key < root.key:
        root.left = _erase(root.left, key)
        _pull(root)
        return root
    elif key > root.key:
        root.right = _erase(root.right, key)
        _pull(root)
        return root
    else:
        return _merge(root.left, root.right)


def _range_add(root, lo_key, hi_key, delta):
    """Add ``delta`` to the weights of all keys in [lo_key, hi_key)."""
    left, mid_right = _split(root, lo_key)
    mid, right = _split(mid_right, hi_key)
    _apply(mid, delta)
    return _merge(left, _merge(mid, right))


def _rank(root, key):
    """Number of keys strictly less than ``key``."""
    r = 0
    n = root
    while n is not None:
        if key <= n.key:
            n = n.left
        else:
            r += _cnt(n.left) + 1
            n = n.right
    return r


def _kth(root, k):
    """Node with rank ``k`` (0-based)."""
    n = root
    while True:
        c = _cnt(n.left)
        if k < c:
            n = n.left
        elif k == c:
            return n
        else:
            k -= c + 1
            n = n.right


def _find_by_weight(root, r):
    """Find the node owning weight slot ``r`` in [0, root.wsum).

    Returns ``(node, residual)`` where ``residual`` is the slot index
    within the node's own weight, in [0, node.w).
    """
    n = root
    while True:
        _push(n)
        lw = _wsum(n.left)
        if r < lw:
            n = n.left
            continue
        r -= lw
        if r < n.w:
            return n, r
        r -= n.w
        n = n.right


_SIDE_NAMES = {'left': LEFT, 'l': LEFT, 'right': RIGHT, 'r': RIGHT}


def _side_index(side):
    if isinstance(side, str):
        try:
            return _SIDE_NAMES[side.lower()]
        except KeyError:
            pass
    elif side == LEFT or side == RIGHT:
        return side
    raise ValueError('side must be LEFT/RIGHT or "left"/"right", got %r' % (side,))


def _is_side(v):
    return (isinstance(v, str) and v.lower() in _SIDE_NAMES) or v == LEFT or v == RIGHT


class BandJoinIndex(object):
    """Current state of a band join between two tables.

    Rows of both tables are kept by row id; a pair ``(l, r)`` matches when
    ``|l.price - r.price| <= tolerance``.  The index supports inserts,
    deletes and re-pricing (``set`` on an existing id) in ``O(log n)``
    expected time, reports the current number of matching pairs in
    ``O(1)``, and draws uniformly random matching pairs (independent, with
    replacement) in ``O(log n)`` expected time per pair.

    Parameters
    ----------
    tolerance: float
        Maximum absolute price difference for a pair to match.
    seed: int, optional
        Seed for the internal treap randomization (structure only, does
        not affect sampling).

    Examples
    --------
    >>> idx = BandJoinIndex(tolerance=0.5)
    >>> idx.set('left', 'a', 10.0)
    >>> idx.set('right', 'x', 10.2)
    >>> idx.set('right', 'y', 20.0)
    >>> idx.total
    1
    >>> idx.delete('right', 'x')
    >>> idx.total
    0
    """

    def __init__(self, tolerance, seed=None):
        if tolerance < 0:
            raise ValueError('tolerance must be non-negative')
        self.tolerance = tolerance
        self._roots = [None, None]
        self._rows = [dict(), dict()]   # per side: {rid: (price, seq)}
        self._seq = [0, 0]
        self._rng = random.Random(seed)

    def __repr__(self):
        return 'BandJoinIndex(tolerance=%r, left=%d, right=%d, pairs=%d)' % (
            self.tolerance, len(self._rows[LEFT]), len(self._rows[RIGHT]),
            self.total)

    # -- state inspection --------------------------------------------------

    @property
    def total(self):
        """Current number of matching pairs."""
        root = self._roots[LEFT]
        return root.wsum if root is not None else 0

    def row_count(self, side):
        """Number of live rows on one side."""
        return len(self._rows[_side_index(side)])

    def state(self, side):
        """Snapshot of one side as a ``{row_id: price}`` dict."""
        rows = self._rows[_side_index(side)]
        return {rid: price for rid, (price, seq) in rows.items()}

    def price_of(self, side, rid):
        """Current price of a row id."""
        return self._rows[_side_index(side)][rid][0]

    # -- mutation ------------------------------------------------------------

    def set(self, side, rid, price):
        """Insert a row or re-price an existing row id."""
        side = _side_index(side)
        if price is None:
            raise ValueError('price is required for set')
        if rid in self._rows[side]:
            self._remove(side, rid)
        seq = self._seq[side]
        self._seq[side] = seq + 1
        self._rows[side][rid] = (price, seq)
        node = _Node((price, seq), self._rng.getrandbits(64), 0, rid)
        if side == LEFT:
            node.w = node.wsum = self._window_count(RIGHT, price)
            self._roots[LEFT] = _insert(self._roots[LEFT], node)
        else:
            node.w = node.wsum = 1
            self._roots[RIGHT] = _insert(self._roots[RIGHT], node)
            self._shift_left_window(price, +1)

    def delete(self, side, rid):
        """Remove a row by id."""
        side = _side_index(side)
        if rid not in self._rows[side]:
            raise KeyError('unknown row id %r on side %r' % (rid, side))
        self._remove(side, rid)

    def _remove(self, side, rid):
        price, seq = self._rows[side].pop(rid)
        key = (price, seq)
        if side == LEFT:
            self._roots[LEFT] = _erase(self._roots[LEFT], key)
        else:
            self._shift_left_window(price, -1)
            self._roots[RIGHT] = _erase(self._roots[RIGHT], key)

    def _window_count(self, side, price):
        """Number of rows on ``side`` with price within tolerance."""
        root = self._roots[side]
        tol = self.tolerance
        return _rank(root, (price + tol, _INF)) - _rank(root, (price - tol, -1))

    def _shift_left_window(self, price, delta):
        """Add ``delta`` to the degree of every left row matching ``price``."""
        tol = self.tolerance
        self._roots[LEFT] = _range_add(self._roots[LEFT],
                                       (price - tol, -1), (price + tol, _INF),
                                       delta)

    # -- sampling ------------------------------------------------------------

    def sample_one(self, rng=None):
        """Draw one matching pair uniformly at random.

        Every currently matching pair has probability ``1 / total``.
        """
        total = self.total
        if total == 0:
            raise ValueError('cannot sample: no matching pairs')
        if rng is None:
            rng = random
        r = rng.randrange(total)
        node, residual = _find_by_weight(self._roots[LEFT], r)
        lprice = node.key[0]
        right = self._roots[RIGHT]
        start = _rank(right, (lprice - self.tolerance, -1))
        rnode = _kth(right, start + residual)
        return Match(node.rid, lprice, rnode.rid, rnode.key[0])

    def sample(self, k=1, rng=None):
        """Draw ``k`` pairs independently, uniformly, with replacement."""
        return [self.sample_one(rng) for _ in range(k)]


# ---------------------------------------------------------------------------
# Stream operators
# ---------------------------------------------------------------------------

_SET_OPS = {'set', 'upsert', 'insert', 'update'}
_DEL_OPS = {'del', 'delete', 'remove'}


def _parse_change(x, default_side):
    """Normalize a change event to ``(side, kind, rid, price)``."""
    side = default_side
    if isinstance(x, dict):
        op = x.get('op', 'set')
        if 'side' in x:
            side = _side_index(x['side'])
        for key in ('id', 'rid', 'row_id'):
            if key in x:
                rid = x[key]
                break
        else:
            raise ValueError('change event must carry an id/rid/row_id field: %r' % (x,))
        price = x.get('price')
    else:
        parts = tuple(x)
        if len(parts) == 4:
            side = _side_index(parts[0])
            op, rid, price = parts[1], parts[2], parts[3]
        elif len(parts) == 3:
            if _is_side(parts[0]) and isinstance(parts[1], str):
                # (side, op, rid) with implicit delete-style payload
                side = _side_index(parts[0])
                op, rid, price = parts[1], parts[2], None
            else:
                op, rid, price = parts
        elif len(parts) == 2:
            op, rid, price = parts[0], parts[1], None
        else:
            raise ValueError('change event must be a (op, rid[, price]) tuple '
                             'optionally prefixed by a side: %r' % (x,))
    if isinstance(op, str):
        op = op.lower()
    if op in _SET_OPS:
        kind = 'set'
    elif op in _DEL_OPS:
        kind = 'del'
    else:
        raise ValueError('unknown change operation %r' % (op,))
    if kind == 'set' and price is None:
        raise ValueError('price is required for set/upsert events')
    return side, kind, rid, price


def _parse_count(x):
    if x is None:
        return 1
    if isinstance(x, int) and x >= 0:
        return int(x)
    raise TypeError('sample request must be a non-negative int or None, got %r' % (x,))


@Stream.register_api()
class band_join(Stream):
    """Maintain a band join of two change streams and emit the pair total.

    Consumes change events from two upstreams (left and right channel).
    Each event is a ``(op, rid[, price])`` tuple with ``op`` in
    ``{'set', 'upsert', 'insert', 'update'}`` or
    ``{'del', 'delete', 'remove'}``, a dict with ``op``/``id``/``price``
    keys, or a 4-tuple ``(side, op, rid, price)`` naming the side
    explicitly.  ``set`` on an existing row id re-prices the row.

    After every processed change the current number of matching pairs
    (``|left_price - right_price| <= tolerance``) is emitted.  The
    maintained state never expands the join: updates and emissions cost
    ``O(log n)`` expected time.

    Parameters
    ----------
    left, right: Stream
        The two upstream change streams.
    tolerance: float
        Maximum absolute price difference for a pair to match.
    seed: int, optional
        Seed for internal index randomization.

    Attributes
    ----------
    index: BandJoinIndex
        The underlying index; use ``join.index.state('left')`` to inspect
        the current rows and ``join.total`` for the pair count.

    Examples
    --------
    >>> from streamz import Stream
    >>> left, right = Stream(), Stream()
    >>> join = left.band_join(right, tolerance=0.5)
    >>> totals = join.sink_to_list()
    >>> _ = left.emit(('set', 'a', 10.0))
    >>> _ = right.emit(('set', 'x', 10.2))
    >>> _ = right.emit(('set', 'y', 20.0))
    >>> totals
    [0, 1, 1]
    >>> _ = right.emit(('del', 'x'))
    >>> totals
    [0, 1, 1, 0]

    See Also
    --------
    sample_pairs
    """
    str_list = ['tolerance']

    def __init__(self, left, right, tolerance, seed=None, **kwargs):
        stream_name = kwargs.pop('stream_name', None)
        if left is right:
            raise ValueError('left and right must be distinct streams')
        self.tolerance = tolerance
        self.index = BandJoinIndex(tolerance, seed=seed)
        self._left = left
        self._right = right
        Stream.__init__(self, upstreams=[left, right], stream_name=stream_name)

    @property
    def total(self):
        """Current number of matching pairs."""
        return self.index.total

    def state(self, side):
        """Snapshot of one side as a ``{row_id: price}`` dict."""
        return self.index.state(side)

    def sample(self, k=1, rng=None):
        """Draw ``k`` matching pairs uniformly, with replacement."""
        return self.index.sample(k, rng)

    def update(self, x, who=None, metadata=None):
        side = LEFT if who is self._left else RIGHT
        side, kind, rid, price = _parse_change(x, side)
        if kind == 'set':
            self.index.set(side, rid, price)
        else:
            self.index.delete(side, rid)
        return self._emit(self.index.total, metadata=metadata)


@Stream.register_api()
class sample_pairs(Stream):
    """Draw uniformly random matching pairs from a ``band_join`` node.

    Every upstream event requests a batch of samples: emit a non-negative
    integer ``k`` to draw ``k`` pairs, or ``None`` to draw one.  Each
    request emits a list of ``Match`` namedtuples
    ``(left_id, left_price, right_id, right_price)`` drawn independently
    and uniformly with replacement from the pairs matching *at request
    time*; each draw costs ``O(log n)`` expected time.  An empty list is
    emitted when no pairs currently match.

    Parameters
    ----------
    upstream: Stream
        Stream of sample requests (ints or None).
    join: band_join or BandJoinIndex
        The join node (or bare index) to sample from.
    seed: int, optional
        Seed for the sampling random generator.

    Examples
    --------
    >>> from streamz import Stream
    >>> left, right = Stream(), Stream()
    >>> join = left.band_join(right, tolerance=0.5)
    >>> trigger = Stream()
    >>> batches = trigger.sample_pairs(join, seed=1).sink_to_list()
    >>> _ = left.emit(('set', 'a', 10.0))
    >>> _ = right.emit(('set', 'x', 10.2))
    >>> _ = trigger.emit(2)
    >>> len(batches[0])
    2
    >>> batches[0][0]
    Match(left_id='a', left_price=10.0, right_id='x', right_price=10.2)

    See Also
    --------
    band_join
    """
    def __init__(self, upstream, join, seed=None, **kwargs):
        stream_name = kwargs.pop('stream_name', None)
        self.join = join
        self._rng = random.Random(seed)
        Stream.__init__(self, upstream, stream_name=stream_name)

    def update(self, x, who=None, metadata=None):
        k = _parse_count(x)
        index = getattr(self.join, 'index', self.join)
        if k == 0 or index.total == 0:
            return self._emit([], metadata=metadata)
        return self._emit(index.sample(k, self._rng), metadata=metadata)
