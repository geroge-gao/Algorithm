class Solution:
    def maxProfit(self, k, prices) -> int:
        if not prices:
            return 0

        n = len(prices)
        k = min(k, n//2)

        buys = [[float('-inf')] * (k+1) for _ in range(n)]
        sells = [[float('-inf')] * (k+1) for _ in range(n)]
        for i in range(n):
            buys[i][0] = 0
            sells[i][0] = 0

        for i in range(1, n):
            for j in range(1,  k+1):
                buys[i][j] = max(buys[i-1][j], sells[i-1][j-1] - prices[i])
                sells[i][j] = max(buys[i-1][j] + prices[i], sells[i-1][j])

        return max(sells[n-1])



if __name__ == '__main__':
    k = 2
    prices = [3,2,6,5,0,3]
    res = Solution().maxProfit(k, prices)
    print(res)
