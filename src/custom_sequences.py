from motzkin_greedy import optimize_motzkin_greedy
from smooth_walk_greedy import optimize_smooth_walk_greedy
from utils import P_from_d_prefix
from ssd_check import is_distinct_subset_sum_optimized
from sequence_analysis import compare_and_save


print(compare_and_save(d_seed, 35, optimizer=optimize_motzkin_greedy, check_ssd=False))
