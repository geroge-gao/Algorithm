import math
import sys

class Solution:
    """
    动态规划
    """
    def numSquares(self, n: int) -> int:

        square_nums = [i**2 for i in range(int(math.sqrt(n))+1)]

        dp = [sys.maxsize] * (n+1)

        dp[0] = 0

        for i in range(1,n+1):
            for square in square_nums:
                if i < square:
                    break

                dp[i] = min(dp[i], dp[i-square]+1)

        return dp[-1]