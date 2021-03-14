class Solution:
    def maximalSquare(self, matrix) -> int:

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0

        max_len = 0
        row, col = len(matrix), len(matrix[0])

        dp = [[0] * col for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i][j-1], dp[i-1][j-1], dp[i-1][j]) + 1
                    max_len = max(dp[i][j], max_len)

        max_square = max_len * max_len

        return max_square
