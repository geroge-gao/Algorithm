# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from queue import Queue

class Solution:
    def levelOrder(self, root):
        res = []
        if root is None:
            return res

        q = Queue()
        q.put(root)

        while q.qsize() != 0:
            level = []
            size = q.qsize()
            while size > 0:
                root = q.get()
                level.append(root.val)

                if root.left is not None:
                    q.put(root.left)

                if root.right is not None:
                    q.put(root.right)
                size -= 1
            res.append(level)

        return res
