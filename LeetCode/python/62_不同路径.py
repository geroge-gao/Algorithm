class Solution:
    def uniquePaths(self, m, n):

        res = [(n * [0]) for i in range(m)]
        for i in range(m):
            res[i][0] = 1

        for j in range(n):
            res[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                res[i][j] = res[i][j-1] + res[i-1][j]

        return res[m-1][n-1]