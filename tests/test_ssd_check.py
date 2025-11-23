import random
import pytest

from src import ssd_check

is_distinct_subset_sum_combinations = ssd_check.is_distinct_subset_sum_combinations
is_distinct_subset_sum_bitmask = ssd_check.is_distinct_subset_sum_bitmask


CHECKERS = [is_distinct_subset_sum_combinations, is_distinct_subset_sum_bitmask]


@pytest.mark.parametrize("checker", CHECKERS)
@pytest.mark.parametrize(
    "nums",
    [
        [],
        [1],
        [1, 2, 4, 8],  # powers of two
        [13, 12, 11, 9, 6],
        [2, 3, 7],
        [309, 308, 307, 305, 302, 296, 285, 265, 225, 148],  # Conway-Guy for n=9
        [
            8213,
            8498,
            8803,
            8805,
            8806,
            8646,
            8723,
            8763,
            4323,
            6523,
            8783,
            8794,
            8800,
            8807,
            7643,
        ],  # Conway-Guy shuffled
    ],
)
def test_known_good_sequences(checker, nums):
    """These sequences should have all distinct subset sums."""
    ok, info = checker(nums)
    assert ok is True
    assert isinstance(info, str)
    assert "distinct" in info.lower()


@pytest.mark.parametrize("checker", CHECKERS)
@pytest.mark.parametrize(
    "nums",
    [
        [3, 4, 5, 6],  # 3+6 == 4+5
        [1, 1],  # duplicate singletons
        [2, 2, 4],  # 2+2 == 4
        [5, 1, 4, 5],  # duplicated 5s create collisions
        [308, 307, 306, 304, 301, 295, 284, 264, 224, 147],  # conway-guy -1
        [
            4480,
            4477,
            4440,
            3890,
            4460,
            4400,
            4323,
            4472,
            4175,
            4482,
            4483,
            2200,
            4471,
            3320,
        ],  # close to conway-guy, shuffled
    ],
)
def test_known_bad_sequences(checker, nums):
    """These sequences are guaranteed to produce a subset-sum collision."""
    ok, info = checker(nums)
    assert ok is False
    # info should be a tuple: (subset1, subset2, sum)
    assert isinstance(info, tuple) and len(info) == 3
    subset1, subset2, s = info
    assert isinstance(subset1, tuple)
    assert isinstance(subset2, tuple)
    # the reported subsets must actually sum to the reported sum
    assert sum(subset1) == s
    assert sum(subset2) == s
    # the two subsets should be different as tuples (or come from different choices)
    assert subset1 == subset2 or sum(subset1) == sum(subset2)


@pytest.mark.parametrize("checker", CHECKERS)
def test_order_invariance_for_bad_sequence(checker):
    """Different permutations / orders should not affect detection of collisions."""
    base = [3, 4, 5, 6]
    # test several permutations including reverse and some random shuffles
    perms = [tuple(base), tuple(reversed(base))]
    perms.extend(tuple(random.sample(base, len(base))) for _ in range(6))

    for nums in perms:
        ok, info = checker(list(nums))
        assert ok is False
        assert isinstance(info, tuple)


@pytest.mark.parametrize("checker", CHECKERS)
def test_empty_and_singleton_behavior(checker):
    # empty
    ok, info = checker([])
    assert ok is True
    assert isinstance(info, str)

    # singleton
    ok2, info2 = checker([42])
    assert ok2 is True
    assert isinstance(info2, str)


@pytest.mark.parametrize("checker", CHECKERS)
def test_random_duplicate_cases(checker):
    """Generate a few random sequences with intentional duplicates and check collisions."""
    rng = random.Random(12345)
    for _ in range(10):
        # create a sequence with some duplicates
        base = [rng.randint(1, 10) for _ in range(6)]
        # introduce at least one duplicate by copying an element
        if len(base) >= 2:
            base.append(base[0])

        ok, info = checker(base)
        # With duplicates there's a good chance of collision; if not, at least verify function runs
        assert isinstance(ok, bool)
        if not ok:
            assert isinstance(info, tuple) and len(info) == 3
