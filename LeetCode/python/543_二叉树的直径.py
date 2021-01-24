# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    """
    二叉树深度变形：左子树+右子树，关键在于左子树+右子树高度最大的不一定在中间节点
    """
    def __init__(self):
        self.max_depth = 0

    def sumTreeDepth(self, root):
        if not root:
            return 0

        left = self.sumTreeDepth(root.left)
        right = self.sumTreeDepth(root.right)
        current = max(left, right)

        if left + right > self.max_depth:
            self.max_depth = left + right

        return current + 1

    def diameterOfBinaryTree(self, root):

        if not root:
            return 0

        self.sumTreeDepth(root)
        return self.max_depth