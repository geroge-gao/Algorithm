# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from input_utils import Tree

class Solution:
    """
    二叉树遍历常规题
    """

    def __init__(self):
        self.total = 0

    def dfs(self, root):

        if root:
            self.dfs(root.right)
            self.total += root.val
            root.val = self.total
            self.dfs(root.left)

    def convertBST(self, root):

        self.dfs(root)
        return root

