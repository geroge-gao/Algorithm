class Solution:
    def maximalSquare(self, matrix) -> int:

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0

        maxSide = 0
        row, col = len(matrix), len(matrix[0])

        dp = [[0] * col for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j]) + 1
                    maxSide = max(dp[i][j], maxSide)

        maxSquare = maxSide * maxSide

        return maxSquare