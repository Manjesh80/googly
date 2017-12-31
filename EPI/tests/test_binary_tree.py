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
