# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
longest_strictly_increasing_subsequence.py -- longest strictly-increasing
subsequence via patience sort.

Given a sequence, this finds the longest run of values you can pick out -- keeping
their original left-to-right order -- that is strictly increasing. For example,
in [3, 1, 4, 1, 5, 9, 2, 6] one longest strictly-increasing pick is
[1, 4, 5, 9] (length 4). "Strictly" means equal values don't count as increasing,
so the answer for [3, 3] has length 1, not 2.

    subsequence, indices = longest_strictly_increasing_subsequence(sequence)
      * subsequence : the chosen values, in order (one valid longest answer).
      * indices     : {position_in_subsequence: index_in_sequence}, so that
                      sequence[indices[i]] == subsequence[i]. Use this when you
                      need to know WHICH original elements were chosen, not just
                      their values.

Ties: a sequence can have several different longest strictly-increasing
subsequences (all the same length). This function resolves ties
deterministically -- same input, same output. The length is always a genuine
maximum; only which elements are chosen (where the sequence allows a choice) is
fixed by the deterministic tie resolution.

Elements only need to be comparable to each other (support `<` and `>=` against
one another); ints, floats, and tuples all work.

------------------------------------------------------------------------------
How it works (the "patience" card-game picture)
------------------------------------------------------------------------------
Deal the values one at a time onto "piles", left to right. A value goes onto the
leftmost pile whose current top is >= it (replacing that top); if no such pile
exists, it starts a new pile on the right. The number of piles at the end is the
length of the longest strictly-increasing subsequence.

The tops of the piles are always sorted left-to-right, which is why we can find
the right pile with a binary search instead of scanning. To recover the actual
subsequence (not just its length), each value remembers its predecessor: the top
of the pile immediately to its left when it was placed. Following predecessor
links back from the last pile reconstructs the subsequence.

Runs in O(n log n) for the forward pass; reconstruction is O(length).

Implementation notes:
  * We don't store whole piles -- only, per pile, the index of its current
    (smallest-seen) top value. The binary search dereferences that index into
    `sequence` to compare values (a hand-rolled search over the tops, whose
    values are sorted left-to-right by construction).
  * The `smallest_tail_index` list is indexed by small consecutive integers, so
    it is a plain list (fast indexing), not a dict.

Reference:
  * This is the patience-sort variant of longest increasing subsequence: a
    binary search places each element on the leftmost pile whose top is >= it,
    and back-pointers to the previous pile's top are followed in reverse from the
    last pile to reconstruct the subsequence. See
    <https://en.wikipedia.org/wiki/Longest_increasing_subsequence
    (section "Efficient algorithms").
"""

from typing import Any

__all__ = ["longest_strictly_increasing_subsequence"]


def longest_strictly_increasing_subsequence(
    sequence: list[Any],
) -> tuple[list[Any], dict[int, int]]:
    """Find a longest strictly-increasing subsequence of `sequence`.

    Args:
        sequence: The input values. The elements MUST be mutually comparable --
            they need to support ``<`` and ``>=`` against one another (a total order),
            since the algorithm compares them directly. Ints, floats, and tuples
            all satisfy this.

    Returns:
        (subsequence, indices):
          * subsequence: the chosen values in order -- one longest
            strictly-increasing subsequence of the input. Ties are resolved
            deterministically (same input, same output).
          * indices: maps each position in `subsequence` to the index it came
            from in `sequence`, so sequence[indices[i]] == subsequence[i].
    """

    # For each value, the index of the value that precedes it in the growing
    # subsequence (the top of the pile just to its left when it was placed).
    # -1 means "nothing precedes it". Following these links back reconstructs
    # the subsequence. Indexed by position in `sequence` (0..len(sequence)-1).
    predecessor = [-1] * len(sequence)

    # Per pile length, the index (into `sequence`) of that pile's current top --
    # the smallest top value we've been able to achieve for a subsequence of
    # that length so far. Slot 0 is a sentinel for "the empty subsequence".
    smallest_tail_index = [-1]

    pile_count = 0

    for i, value in enumerate(sequence):

        # Find the pile this value belongs on: the leftmost pile (1..pile_count)
        # whose top value is >= value. Because the comparison is ">=", equal
        # values land on an existing pile rather than starting a new one -- this
        # is what makes the result STRICTLY increasing. The pile tops are sorted
        # left-to-right by construction, so a binary search finds the slot.
        lo = 1
        hi = pile_count + 1
        while lo < hi:
            mid = lo + (hi - lo) // 2  # lo <= mid < hi
            if sequence[smallest_tail_index[mid]] >= value:
                hi = mid
            else:
                lo = mid + 1
        pile = lo

        # The value just left of this one in the subsequence is the top of the
        # previous pile.
        predecessor[i] = smallest_tail_index[pile - 1]

        if pile > pile_count:
            # No existing pile could hold it -> it starts a new pile on the right.
            smallest_tail_index.append(i)
            pile_count = pile
        else:
            # It replaces the top of an existing pile, giving that pile length a
            # smaller (better) top to build on later.
            smallest_tail_index[pile] = i

    # ---------------- reconstruction ----------------
    # Walk the predecessor chain back from the top of the last pile, filling the
    # subsequence right-to-left.
    subsequence = [None] * pile_count
    indices = {}
    node = smallest_tail_index[pile_count]
    for position in range(pile_count - 1, -1, -1):
        indices[position] = node
        subsequence[position] = sequence[node]
        node = predecessor[node]

    return subsequence, indices
