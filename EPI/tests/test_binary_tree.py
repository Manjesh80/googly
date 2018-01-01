from ..binary_trees import *


# 9.1 Test if a binary tree is height balanced
# pytest EPI\tests\test_binary_tree.py::test_calculate_tree_depth
def test_calculate_tree_depth():
    root = build_tree()
    height_of_left_tree = calculate_tree_depth(root.left)
    height_of_right_tree = calculate_tree_depth(root.right)
    assert abs(height_of_left_tree - height_of_right_tree) < 2


# 9.2 Check if a tree is Symmetric
# pytest EPI\tests\test_binary_tree.py::test_is_tree_symmetric
def test_is_tree_symmetric():
    root = build_symmetric_tree()
    assert True == is_tree_symmetric(root)


# 9.4 Compute the LCA with Parent( lowest common ancestor ) in Binary Tree
# pytest EPI\tests\test_binary_tree.py::test_lca_with_parent
def test_lca_with_parent():
    tree = build_tree_dict_with_parent()
    parent = lca_with_parent(tree['m'], tree['p'])
    assert parent.data == 'i'


# 9.6 Match sum for the path
# pytest EPI\tests\test_binary_tree.py::sum_the_root_to_leaf_path
def test_match_sum():
    tree = build_tree_dict()
    parent = sum_the_root_to_leaf_path(tree['a'])
    parent = match_sum(tree['a'], 591)
    assert len(parent) == 2


# 9.7 Implement an inorder traversal without recursion
# pytest -s EPI\tests\test_binary_tree.py::test_traverse_in_order_stack
def test_traverse_in_order_stack():
    res = traverse_in_order_stack(build_tree_dict()['a'])
    assert len(res) == 16
