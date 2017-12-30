from ..binary_trees import *


# 9.1 Test if a binary tree is height balanced
# pytest EPI\tests\test_binary_tree.py::test_calculate_tree_depth
def test_calculate_tree_depth():
    root = build_tree()
    height_of_left_tree = calculate_tree_depth(root.left)
    height_of_right_tree = calculate_tree_depth(root.right)
    assert abs(height_of_left_tree - height_of_right_tree) < 2
