# Motzkin–Greedy Constructions for Erdős' Distinct Subset Sums Problem

This repository contains code and computational data for exploring a new family
of constructions for the *distinct subset sums* (DSS) problem of Erdős.

The classical candidate construction is the **Conway–Guy** sequence.  
In this project, I introduce a new **Motzkin–greedy** construction in the
*difference–vector space* of a DSS set, based on **weighted Motzkin path areas**.
This yields:

- An alternative combinatorial interpretation of Conway–Guy (seed $(1,1)$).
- Several **new infinite candidate families**, depending on a seed $(d_1,d_2)$.
- A particularly striking family (seed $(2,1)$) that produces DSS sets
  with maximal elements **smaller** than Conway–Guy for  
  $n = 12, 13, ..., 18$ (verified via full brute-force checks).

All computations here are fully reproducible and all code is open.

## 📌 The Motzkin–Greedy Construction (summary)

Given initial seed $(d(1), d(2))$, define recursively:

1. Generate all truncated Motzkin height profiles $z \in S'(n)$.
2. Compute weighted areas  
   $$F(n) = \{ \sum_i z(i) \cdot d(i) : z \in S'(n) \}$$
3. Set  
   $$d(n) = \text{smallest positive integer NOT in } F(n)$$

4. For each prefix $(d(1),...,d(n))$, construct the decreasing DSS candidate set 
   
   $$P_n = [\sum_{j=k}^n d(j)]_{k=1..n}$$

A set $P_n$ is DSS iff all subset sums are distinct.  
We verify this using **full exhaustive brute-force enumeration**.

## 🚀 Main discovery: Seed $(2,1)$ beats Conway–Guy

For seed $(2,1)$, the Motzkin–greedy rule yields difference vectors:

$$d = 2, 1, 1, 4, 6, 11, 22, 39, 78, 150, 289, 556, 1112, 2185, 4292$$

The corresponding $P_n$ are:

- **DSS for all $n \leq 18$**, and  
- have **strictly smaller maximum** than Conway–Guy for   **$n = 12–18$**.

This is surprising because:

- Until recently, optimal values were known only for $n \leq 9$.  
- A construction improving Conway–Guy at *any* specific $n$ is already rare.  
- This construction improves at **seven consecutive values**.

All data and verification logs are included in `/data` and `/results`.

## 🧪 Reproducibility

To generate some sequences (parallel cores):

```bash
python src/generate_sequences.py --workers 3 --seeds 1,2 --max-n 27 --check-ssd
python src/generate_sequences.py --workers 3 --seeds 3,4 3,5 3,6 6,1 6,2 6,3 6,4  --max-n 22 --check-ssd
```

To generate default sequences:

```bash
python src/generate_sequences.py --workers 13 --max-n 25
```

To test DSS:

```bash
python src/ssd_check.py data/P_sets_seed_2_1.json
```

Two independent brute-force checkers are provided:

- combinatorial subsets (via itertools.combinations)
- bitmask enumeration
