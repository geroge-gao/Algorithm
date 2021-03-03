class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])

        visited = [[False] * n for _ in range(m)]

        rows = [-1, 0, 1, 0]
        cols = [0, 1, 0, -1]

        def dfs(x, y, idx):
            """搜索单词
            Args:
                x: 行索引
                y: 列索引
                idx: 单词对应的字母索引
            """
            if board[x][y] != word[idx]:
                return False

            if idx == len(word) - 1:
                return True

            # 先标记
            visited[x][y] = True

            # 找到符合的字母时开始向四个方向扩散搜索
            for i in range(4):
                nx = x + rows[i]
                ny = y + cols[i]
                if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and dfs(nx, ny, idx + 1):
                    return True
            # 扩散未搜索对应的字母，释放标记
            # 继续往其他方位搜索
            visited[x][y] = False
            return False

        for x in range(m):
            for y in range(n):
                if dfs(x, y, 0):
                    return True

        return False