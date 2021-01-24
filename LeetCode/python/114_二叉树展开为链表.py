# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def __init__(self):
        self.result = []

    def preorder(self, root):
        if root:
            self.result.append(root)
            self.preorder(root.left)
            self.preorder(root.right)

    def flatten(self, root):
        """
        Do not return anything, modify root in-place instead.
        """
        self.preorder(root)

        for i in range(len(self.result)-1):
            self.result[i].right = self.result[i+1]

        for i in range(len(self.result)):
            self.result[i].left = None

        return root