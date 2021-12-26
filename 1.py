class Solution:
    """
    前缀和+递归，与560题思路类似，注意
    """
    # def pathSum(self, root: TreeNode, sum: int) -> int:
    #     #
    #     #     prefit = {0:1}
    #     #     count = 0
    #     #
    #     #     def dfs(root, total, prefix):
    #     #         nonlocal sum, count
    #     #         if root:
    #     #             total += root.val
    #     #
    #     #             if total - sum in prefix:
    #     #                 total += prefix[total - sum]
    #     #
    #     #             if total in prefix:
    #     #                 prefix[total] += 1
    #     #             else:
    #     #                 prefix[total] = 1
    #     #
    #     #             dfs(root.left, total, prefix)
    #     #             dfs(root.right, total, prefix)
    #     #
    #     #             prefix[total] -= 1
    #     #             total -= root.val

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def match(i, j):
            if i == 0:
                return False

            if p[j-1] == '.':
                return True

            return s[i] == p[j]

        dp = [[False] * (n+1) for _ in range(m+1)]

        for i in range(m+1):
            for j in range(1, n+1):

                if p[j-1] == '*':
                    dp[i][j] |= dp[i][j-2]
                    if match(i, j-1):
                        dp[i][j] |= dp[i-1][j]
                else:
                    if match(i, j):
                        dp[i][j] = dp[i-1][j-1]
        return dp[m][n]

import tensorflow.keras.utils