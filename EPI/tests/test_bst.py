from ..bst import *


# 14 Search in BST
# pytest -s EPI\tests\test_bst.py::test_search_bst
def test_search_bst():
    root = build_tree()
    assert search_bst(root, 23) == 23
    assert search_bst(root, 100) is None


# 14.1 Is tree a BST
# pytest -s EPI\tests\test_bst.py::test_is_tree_bst
def test_is_tree_bst():
    root = build_tree()
    res = is_tree_bst_new(root)
    assert res == True


# 14.2 Search and next did not understand
# TODO : Could not understand
# pytest -s EPI\tests\test_bst.py::test_search_and_next_epi
def test_search_and_next_epi():
    root = build_tree()
    res = search_and_next_epi(root, 5)
    assert res.data == 7


# 14.3 Find k largest elements
# pytest -s EPI\tests\test_bst.py::test_reverse_order_traversal
def test_reverse_order_traversal():
    root = build_tree()
    res = reverse_order_traversal(root, 5)
    assert len(res) == 5


# 14.4 Compute LCA in a BST
# pytest -s EPI\tests\test_bst.py::test_compute_lca_bst
def test_compute_lca_bst():
    nodes = build_tree_nodes()
    res = compute_lca_bst(nodes['A'], nodes['J'], nodes['P'])
    assert res.name == 'I'


# 14.5 Reconstruct a tree from a traversal data
# pytest -s EPI\tests\test_bst.py::test_reconstruct_bst_from_traversal_data
def test_reconstruct_bst_from_traversal_data():
    nodes = pre_order_traversal(build_tree_nodes()['A'])
    res = reconstruct_bst_from_traversal_data(nodes)
    sorted_array = in_order_traversal_emit(res)
    assert sorted_array[0] == 2
    assert sorted_array[-1:][0] == 53


# 14.6 Find closest entry in three sorted arrays
# pytest -s EPI\tests\test_bst.py::test_closest_entry_in_3_arrays
def test_closest_entry_in_3_arrays():
    res = closest_entry_in_3_arrays([[1, 2, 10], [3, 4, 5], [4, 5, 6]])
    assert res == 2


# 14.9 Build a balances BST
# pytest -s EPI\tests\test_bst.py::test_build_a_balanced_sbt
def test_build_a_balanced_sbt():
    root = build_a_balanced_sbt([1, 2, 3, 4, 5, 6, 7])
    assert root.data == 4
    assert root.left.data == 2
    assert root.right.data == 6


# 14.10 Delete a node in BST
# pytest -s EPI\tests\test_bst.py::test_delete_node_in_bst
def test_delete_node_in_bst():
    # root = build_a_balanced_sbt([*range(1, 16)])
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 1)) \
           == [2, 3, 4, 5, 6, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 2)) \
           == [1, 3, 4, 5, 6, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 3)) \
           == [1, 2, 4, 5, 6, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 4)) \
           == [1, 2, 3, 5, 6, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 5)) \
           == [1, 2, 3, 4, 6, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 6)) \
           == [1, 2, 3, 4, 5, 7]
    assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 7)) \
           == [1, 2, 3, 4, 5, 6]
    # assert in_order_traversal_emit(delete_node_in_bst(build_a_balanced_sbt([*range(1, 8)]), 8)) \
    #        == [1, 2, 3, 4, 5, 6, 7]


# 14.11 Test if 3 BST nodes are totally ordered
# pytest -s EPI\tests\test_bst.py::test_are_3_nodes_orderes
def test_are_3_nodes_orderes():
    nodes = build_tree_nodes()
    assert are_3_nodes_orderes(nodes['A'], nodes['J'], nodes['K']) == True
    assert are_3_nodes_orderes(nodes['I'], nodes['J'], nodes['M']) == True
    assert are_3_nodes_orderes(nodes['I'], nodes['J'], nodes['P']) == False
    assert are_3_nodes_orderes(nodes['J'], nodes['J'], nodes['K']) == False

# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
# Comment to move file to top
