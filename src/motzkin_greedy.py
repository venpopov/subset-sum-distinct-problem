"""
Motzkin–greedy construction for distinct subset sums.

Core responsibilities:
- Generate truncated Motzkin height profiles S'(n)
- Compute F(n) of weighted areas
- Compute the Motzkin–greedy difference sequence d(n)
"""

from functools import lru_cache
from typing import List, Tuple, Dict


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
