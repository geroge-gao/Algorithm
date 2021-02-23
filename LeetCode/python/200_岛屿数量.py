class Solution:
    def numIslands(self, grid):

        m = len(grid)
        n = len(grid[0])

        visited = [[0] * n for _ in range(m)]

        from collections import deque
        q = deque()
        count = 0
        for i in range(m):
            for j in range(n):
                flag = False
                if not q and grid[i][j] == '1' and visited[i][j] == 0:
                    q.append((i, j))
                    visited[i][j] = 1

                while q:
                    pos = q[0]
                    x, y = pos[0], pos[1]

                    # 向左移动
                    if x - 1 >= 0:
                        if grid[x-1][y] == '1' and visited[x-1][y] == 0:
                            q.append((x-1, y))
                            visited[x-1][y] = 1

                    # 向上移动
                    if y - 1 >= 0:
                        if grid[x][y-1] == '1' and visited[x][y-1] == 0:
                            q.append((x, y-1))
                            visited[x][y-1] = 1

                    # 向下移动
                    if x + 1 < m:
                        if grid[x+1][y] == '1' and visited[x+1][y] == 0:
                            q.append((x+1, y))
                            visited[x+1][y] = 1

                    # 向右移动
                    if y + 1 < n:
                        if grid[x][y+1] == '1' and visited[x][y+1] == 0:
                            q.append((x, y+1))
                            visited[x][y+1] = 1

                    q.popleft()
                    flag = True

                if flag:
                    count += 1

        return count
