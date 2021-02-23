class Solution:
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])

        if m == 0:
            return 0

        # 这种写法存在浅拷贝的坑
        # min_path = [[0] * n] * m

        min_path = [([0] * n) for i in range(m)]
        # 初始化第一行与第一列
        init_dis = 0
        for i in range(m):
            init_dis += grid[i][0]
            min_path[i][0] = init_dis

        init_dis = grid[0][0]
        for i in range(1, n):
            init_dis += grid[0][i]
            min_path[0][i] += init_dis

        for i in range(1, m):
            for j in range(1, n):
                min_path[i][j] = min(min_path[i-1][j], min_path[i][j-1]) + grid[i][j]

        return min_path[m-1][n-1]