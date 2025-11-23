from functools import lru_cache

# ---------- Motzkin profiles ----------


def motzkin_trunc_profiles(n):
    """
    Return all truncated Motzkin height profiles z in S'(n):
    h_0,...,h_{n-1} with:
      - h_0 = h_{n-1} = 0
      - h_i >= 0
      - |h_{i+1} - h_i| in { -1, 0, 1 }
    and z = (h_1,...,h_{n-2}), excluding the completely flat path.
    """
    res = []

    def rec(pos, h, hs):
        # pos: index 0..n-1, h: current height, hs: height list
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
            # must be able to get back down to 0 in "remaining" steps
            if nh > remaining:
                continue
            rec(pos + 1, nh, hs + [nh])

    rec(0, 0, [0])
    return res


@lru_cache(None)
def motzkin_trunc_profiles_cached(n):
    return motzkin_trunc_profiles(n)


# ---------- Motzkin-greedy for a given seed ----------


def motzkin_greedy(max_n, d1, d2):
    """
    Compute d(1)..d(max_n) via the Motzkin-greedy rule:
      d(1) = d1, d(2) = d2
      for n > 2:
        F(n) = { sum_{i=1}^{n-2} z(i)*d(i) : z in S'(n) }
        d(n) = smallest positive integer not in F(n)
    """
    d = {1: d1}
    if max_n >= 2:
        d[2] = d2

    for n in range(3, max_n + 1):
        z_list = motzkin_trunc_profiles_cached(n)
        F = set()
        for z in z_list:
            s = 0
            # z has length n-2; indices i = 1..n-2
            for i, zi in enumerate(z, start=1):
                s += zi * d[i]
            if s > 0:
                F.add(s)
        k = 1
        while k in F:
            k += 1
        d[n] = k

    return [d[i] for i in range(1, max_n + 1)]


# ---------- P from d, and SSD checker ----------


def P_from_d_prefix(d_prefix):
    """
    Given d_1..d_n, construct decreasing P:
      p_k = sum_{j=k}^n d_j
    """
    s = 0
    p_rev = []
    for dj in reversed(d_prefix):
        s += dj
        p_rev.append(s)
    return list(reversed(p_rev))  # [p_1,...,p_n]


# ---------- Experiment driver ----------


def analyze_seed(seed, max_n):
    d1, d2 = seed
    print(f"\n=== Seed {seed}, max_n={max_n} ===")
    d_seq = motzkin_greedy(max_n, d1, d2)
    print("d-sequence:")
    print(d_seq)

    ssd_flags = []
    maxP_list = []
    first_fail = None

    for n in range(1, max_n + 1):
        Pn = P_from_d_prefix(d_seq[:n])
        ok, info = is_distinct_subset_sum(Pn)
        ssd_flags.append(1 if ok else 0)
        maxP_list.append(Pn[0])
        if not ok and first_fail is None:
            first_fail = n
        if not ok:
            print(f"n={n}: NOT DSS, collision: {info}")
        # Uncomment to see all Pn:
        # print(f"n={n}: P={Pn}, SSD={ok}")

    print("SSD flags (1=DSS, 0=collision) for n=1..max_n:")
    print(ssd_flags)
    print("max P_n for n=1..max_n:")
    print(maxP_list)
    if first_fail is None:
        print("No SSD failure up to n =", max_n)
    else:
        print("First SSD failure at n =", first_fail)


if __name__ == "__main__":
    max_n = 20  # adjust if you want to push higher
    seeds = [
        (3, 2),  # extend this one
        (2, 2),
        (4, 1),
        (4, 2),
        (1, 4),
        (3, 3),  # my extra pick
    ]
    for seed in seeds:
        analyze_seed(seed, max_n)
