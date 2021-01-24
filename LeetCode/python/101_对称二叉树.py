# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def compare(self, p, q):
        if p is None and q is None:
            return True

        if p is None or q is None:
            return False

        return p.val == q.val and self.compare(p.right, q.left) and self.compare(p.left, q.right)

    def isSymmetric(self, root):
        return self.compare(root, root)

