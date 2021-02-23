
class Solution:
    def exist(self, board, word) -> bool:

        m = len(board)
        n = len(board[0])
        length = len(word)


        # 定义一个栈
        stack = []
        for i in range(m):
            for j in range(n):
                count = 0
                flag = [([False] * n) for _ in range(m)]
                if board[i][j] == word[0] and not flag[i][j]:
                    stack.append((i, j))
                    flag[i][j] = True
                    count += 1
                while stack and count < length:

                    x, y = stack[-1]
                    f = False

                    if x - 1 >= 0 and not flag[x-1][y] and board[x-1][y] == word[count]:
                        stack.append((x-1, y))
                        flag[x-1][y] = True
                        count += 1
                        f = True

                    elif x + 1 < m and not flag[x+1][y] and board[x+1][y] == word[count]:
                        stack.append((x+1, y))
                        flag[x+1][y] = True
                        count += 1
                        f = True

                    elif y - 1 >= 0 and not flag[x][y-1] and board[x][y-1] == word[count]:
                        stack.append((x, y-1))
                        flag[x][y-1] = True
                        count += 1
                        f = True

                    elif y + 1 < n and not flag[x][y+1] and board[x][y+1] == word[count]:
                        stack.append((x, y+1))
                        flag[x][y+1] = True
                        count += 1
                        f = True

                    if not f:
                        stack.pop()
                        count -= 1

                if count == length:
                    return True

        return False