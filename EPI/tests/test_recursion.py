from ..recursion import *


# 15.1 Tower of Hanoi
# pytest EPI\tests\test_recursion.py::test_compute_tower_of_hanoi
def test_compute_tower_of_hanoi():
    pass


# 15.2 N Queen Placement
# pytest EPI\tests\test_recursion.py::test_n_queen_placement
def test_n_queen_placement():
    res = n_queen_placement_epi(4)
    assert len(res) == 2
    assert res[0] == [1, 3, 0, 2]
    assert res[1] == [2, 0, 3, 1]


# 15.3 Generate permutations
# pytest EPI\tests\test_recursion.py::test_permutations_epi
def test_permutations_epi():
    res = permutations_epi_manjesh([1, 2, 3])
    assert len(res) == 6


# 15.4 Generate the power set
# pytest EPI\tests\test_recursion.py::test_generate_power_subset
def test_generate_power_subset():
    res = generate_power_subset([1, 2])
    assert len(res) == 4


# 15.5 Generate K subset
# pytest -s EPI\tests\test_recursion.py::test_generate_k_subset
def test_generate_k_subset():
    res = generate_k_subset_new([1, 2, 3, 4], 2)
    assert len(res) == 6
