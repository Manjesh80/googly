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
