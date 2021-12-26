class Solution:
    def coinChange(self, coins, amount) -> int:
        import sys
        n = len(coins)
        dp =[sys.maxsize] * (amount+1)
        dp[0]=0
        coins.sort()
        for i in range(1, amount+1):
            for j in coins:
                if i < j:
                    break
                dp[i] = min(dp[i-j]+1, dp[i])

        return dp[amount] if dp[amount] != sys.maxsize else -1


if __name__ == '__main__':
    coins = [474,83,404,3]
    amount = 264
    res = Solution().coinChange(coins, amount)
    print(res)
