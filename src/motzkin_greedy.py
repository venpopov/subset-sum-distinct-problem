"""
Motzkin–greedy construction for distinct subset sums.

Core responsibilities:
- Generate truncated Motzkin height profiles S'(n)
- Compute F(n) of weighted areas
- Compute the Motzkin–greedy difference sequence d(n)
"""

from functools import lru_cache
from typing import List, Tuple, Dict, Set


def _motzkin_trunc_profiles(n: int) -> List[Tuple[int, ...]]:
    """
    Generate all truncated Motzkin height profiles z in S'(n).

    A full profile h = (h_0,...,h_{n-1}) satisfies:
        - h_0 = h_{n-1} = 0
        - h_i >= 0
        - |h_{i+1} - h_i| in { -1, 0, 1 }

    We return z = (h_1,...,h_{n-2}) and exclude the completely flat path.
    """
    res: List[Tuple[int, ...]] = []

    def rec(pos: int, h: int, hs: List[int]) -> None:
        """
        pos: current index in 0..n-1
        h:   current height
        hs:  list of heights [h_0,...,h_pos]
        """
        if pos == n - 1:
            # must return to 0
            if h == 0:
                z = hs[1:-1]
                if any(z):  # exclude flat path
                    res.append(tuple(z))
            return

        remaining = (n - 1) - pos  # steps left

        for delta in (-1, 0, 1):
            nh = h + delta
            if nh < 0:
                continue
            # must be able to get back to 0 in "remaining" steps with step size 1
            if nh > remaining:
                continue
            rec(pos + 1, nh, hs + [nh])

    rec(0, 0, [0])
    return res


@lru_cache(None)
def motzkin_trunc_profiles(n: int) -> List[Tuple[int, ...]]:
    """
    Cached wrapper for truncated Motzkin profiles S'(n).
    """
    return _motzkin_trunc_profiles(n)


def motzkin_greedy(max_n: int, d1: int, d2: int) -> List[int]:
    """
    Compute the Motzkin–greedy difference sequence d(1)..d(max_n)
    with initial seed (d(1), d(2)) = (d1, d2).

    Recurrence:
        For n > 2:
            F(n) = { sum_{i=1}^{n-2} z(i)*d(i) : z in S'(n) }
            d(n) = smallest positive integer NOT in F(n)

    Returns:
        d_seq: list [d(1), ..., d(max_n)]
    """
    if max_n < 1:
        return []

    d: Dict[int, int] = {1: d1}
    if max_n >= 2:
        d[2] = d2

    for n in range(3, max_n + 1):
        z_list = motzkin_trunc_profiles(n)
        F = set()

        for z in z_list:
            s = 0
            # z has length n-2, indices i = 1..n-2
            for i, zi in enumerate(z, start=1):
                s += zi * d[i]
            if s > 0:
                F.add(s)

        k = 1
        while k in F:
            k += 1
        d[n] = k

    return [d[i] for i in range(1, max_n + 1)]


## v2


def _compute_F_for_n(n: int, d: List[int]) -> Set[int]:
    """
    Compute F(n) = { sum_{i=1}^{n-2} h_i * d[i] : h is a Motzkin profile of length n }
    where:
      - h_0 = h_{n-1} = 0
      - h_i >= 0
      - |h_{i+1} - h_i| in {-1, 0, 1}
      - we exclude the completely flat path (all h_i = 0)
    and we only consider indices 1 <= i <= n-2 in the sum.

    Arguments:
        n: length of the full height profile (h_0,...,h_{n-1})
        d: 1-based list of differences, d[1..n-1] must be valid

    Returns:
        A set F(n) of all positive weighted areas > 0.
    """
    F: Set[int] = set()

    # Depth-first search over Motzkin height profiles
    def dfs(pos: int, h: int, s: int, ever_positive: bool) -> None:
        """
        pos: current position in 0..n-1, at height h = h_pos
        s:   current weighted sum sum_{i=1}^pos h_i * d[i] (only 1 <= i <= n-2)
        ever_positive: True if any h_i > 0 for 1 <= i <= n-2 so far
        """
        if pos == n - 1:
            # Must return to 0 at the end; exclude completely flat path
            if h == 0 and ever_positive:
                if s > 0:
                    F.add(s)
            return

        remaining = (n - 1) - pos  # steps left to reach position n-1

        # Next position
        next_pos = pos + 1

        for delta in (-1, 0, 1):
            nh = h + delta
            if nh < 0:
                continue
            # must be able to get back to 0 in "remaining" steps of step size 1
            if nh > remaining:
                continue

            new_s = s
            new_ever = ever_positive

            # We only accumulate weighted area for indices 1..n-2
            if 1 <= next_pos <= n - 2:
                if nh > 0:
                    new_ever = True
                new_s += nh * d[next_pos]

            dfs(next_pos, nh, new_s, new_ever)

    # Start at position 0, height 0, sum 0, never positive yet
    dfs(pos=0, h=0, s=0, ever_positive=False)
    return F


def motzkin_greedy_v2(max_n: int, d1: int, d2: int) -> List[int]:
    """
    Compute the Motzkin–greedy difference sequence d(1)..d(max_n)
    with initial seed (d(1), d(2)) = (d1, d2).

    Recurrence:
        For n > 2:
            Let F(n) be the set of weighted areas
                F(n) = { sum_{i=1}^{n-2} h_i * d(i) : h is a Motzkin profile of length n }
            excluding the flat profile.

            Then
                d(n) = smallest positive integer not in F(n).

    Implementation details:
        - We keep d as a 1-based list of length max_n+1:
              d[1] = d1, d[2] = d2, ...
        - For each n, _compute_F_for_n(n, d) explores all Motzkin height profiles
          of length n and returns the set of weighted sums F(n).

    Returns:
        d_seq: list [d(1), ..., d(max_n)]
    """
    if max_n < 1:
        return []

    # 1-based indexing for d: d[1], d[2], ..., d[max_n]
    d = [0] * (max_n + 1)
    d[1] = d1
    if max_n >= 2:
        d[2] = d2

    for n in range(3, max_n + 1):
        F_n = _compute_F_for_n(n, d)
        k = 1
        while k in F_n:
            k += 1
        d[n] = k

    return d[1 : max_n + 1]


if __name__ == "__main__":
    # Tiny sanity check: seed (1,1) should match Conway–Guy differences
    print(motzkin_greedy(10, 1, 1))
    print(motzkin_greedy_v2(10, 1, 1))
