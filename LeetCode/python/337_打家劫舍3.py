# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:

    """
    思路一：dfs+深度优先计算
    """

    def rob(self, root) -> int:

        def dfs(root):

            if not root:
                return 0, 0

            left = dfs(root.left)
            right = dfs(root.right)
            rob_value = left[1] + right[1] + root.val
            skip_value = max(left[0], left[1]) + max(right[0], right[1])
            return [rob_value, skip_value]

        value = dfs(root)

        return max(value[0], value[1])
