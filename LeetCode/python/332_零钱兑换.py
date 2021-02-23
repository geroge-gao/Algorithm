class Solution:
    def coinChange(self, coins, amount):
        import sys
        n = len(coins)
        dp = [sys.maxsize] * (amount+1)
        dp[0] = 0

        for i in coins:
            for j in range(i, amount+1):
                dp[j] = min(dp[j], dp[j-i]+1)

        return dp[amount] if dp[amount] != sys.maxsize else -1
