from ..hash_tables import *
from collections import *
from ..binary_trees import *


class Employee:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return True
        # return self.name == other.name

    def __hash__(self):
        return 1


# pytest -s EPI\tests\test_hash_tables.py::test_bad_hash
def test_bad_hash():
    e1 = Employee("Ganesh")
    e2 = Employee("Suresh")
    d = {e1: e1, e2: e2}
    assert d[e1].name != "Ganesh"
    assert d[e2].name == "Suresh"


# pytest -s EPI\tests\test_hash_tables.py::test_collections
def test_collections():
    l = ['A', 'B', 'A', 'C', 'D', 'C', ]
    c = Counter(l)
    print(c)
    print(c['A'])
    print(' *********************** ')
    print(Counter('A') + Counter('A'))
    print(Counter('A') - Counter('A'))
    print(Counter(a=3, b=1) & Counter(b=2))
    # print(Counter('A'=3, 'B'=1) & Counter('b'=3))
    print(Counter('A') | Counter('B'))
    print(' *********************** ')


# 12.1 Test if a value can be permuted for palindromcity
# pytest -s EPI\tests\test_hash_tables.py::test_can_string_be_permuted_for_palindromicity
def test_can_string_be_permuted_for_palindromicity():
    assert True == can_string_be_permuted_for_palindromicity('level')
    assert False == can_string_be_permuted_for_palindromicity('Ganesh')


# 12.3 Implement ISBN LRU Search
# pytest -s EPI\tests\test_hash_tables.py::test_LRU_cache
def test_LRU_cache():
    lru = LRUCache(3)
    lru.insert('A', 1)
    lru.insert('B', 2)
    lru.insert('C', 3)
    lru.insert('A', 2)
    lru.insert('D', 4)
    lru.insert('E', 4)
    assert lru.lookup('B')[0] == False
    assert lru.lookup('C')[0] == False
    assert lru.lookup('A')[0] == True


# 12.4 Compute the LCA with Parent( lowest common ancestor ) in Binary Tree
# pytest -s EPI\tests\test_hash_tables.py::test_lca_with_parent_optimized
def test_lca_with_parent_optimized():
    tree = build_tree_dict_with_parent()
    parent = lca_with_parent_optimized(tree['m'], tree['p'])
    assert parent.name == 'i'


# 12.5 Compute K most frequent queries
# pytest -s EPI\tests\test_hash_tables.py::test_k_most_frequent_queries
def test_k_most_frequent_queries():
    queries = ['Query1', 'Query2', 'Query3', 'Query4', 'Query1', 'Query2', 'Query2',
               'Query1', 'Query4', 'Query1', 'Query4', 'Query1', 'Query4', 'Query1',
               'Query1', 'Query4', 'Query1', 'Query4', 'Query1', 'Query4', 'Query1']
    res = k_most_frequent_queries(queries, 2)
    assert res[0][1] == 'Query4'
    assert res[1][1] == 'Query1'


# 12.7 Find the smallest Sub Array covering all values
# pytest -s EPI\tests\test_hash_tables.py::test_find_smallest_subarray_covering_subset_with_odict
def test_find_smallest_subarray_covering_subset_with_odict():
    stream = [s for s in 'deefdef']
    query_strings = ['d', 'e', 'f']
    res = find_smallest_subarray_covering_subset_with_odict(stream, query_strings)
    assert res[0] == 2
    assert res[1] == 4


# 12.8 Find the smallest Sub Array sequentially covering all values
# pytest -s EPI\tests\test_hash_tables.py::test_find_smallest_subarray_sequentially_covering_subset_with_odict
def test_find_smallest_subarray_sequentially_covering_subset_with_odict():
    stream = [s for s in 'abcefefadeefdefaac']
    stream = [s for s in 'deeasdfdabfdeedf']
    query_strings = ['d', 'e', 'f']
    res = find_smallest_subarray_sequentially_covering_subset_with_odict(stream, query_strings)
    print(res)
    assert res[0] == 0
    assert res[1] == 6


# 12.9 Find longest sub-array with distinct entries
# pytest -s EPI\tests\test_hash_tables.py::test_find_the_longest_sub_array_with_distinct_entries
def test_find_the_longest_sub_array_with_distinct_entries():
    stream = [s for s in 'abacdac']
    res = find_the_longest_sub_array_with_distinct_entries(stream)
    assert res[0] == 1
    assert res[1] == 5


# 12.10 Find the length of the longest contained interval
# pytest -s EPI\tests\test_hash_tables.py::test_len_of_longest_contained_interval
def test_len_of_longest_contained_interval():
    nums = [3, -2, 7, 9, 8, 1, 2, 0, -1, 5, 8]
    assert 6 == len_of_longest_contained_interval(nums)


# 12.11 Find the student with top three scores
# pytest -s EPI\tests\test_hash_tables.py::test_top_three_scores
def test_top_three_scores():
    scores = ['Ganesh,100', 'Ganesh,100', 'Ganesh,100', 'Ganesh,100', 'Ganesh,100', 'Ganesh,100', 'Ganesh,100',
              'Ganesh1,10', 'Ganesh2,20', 'Ganesh3,30', 'Ganesh4,40', 'Ganesh5,50', 'Ganesh6,60', 'Ganesh7,70',
              'Ganesh1,10', 'Ganesh2,20', 'Ganesh3,30', 'Ganesh4,40', 'Ganesh5,50', 'Ganesh6,60', 'Ganesh7,70',
              'Ganesh1,10', 'Ganesh2,20', 'Ganesh3,30', 'Ganesh4,40', 'Ganesh5,50', 'Ganesh6,60', 'Ganesh7,70',
              'Ganesh1,10', 'Ganesh2,20', 'Ganesh3,30', 'Ganesh4,40', 'Ganesh5,50', 'Ganesh6,60', 'Ganesh7,70']
    assert 300 == top_three_scores(scores)
