class Solution:
    def generateParenthesis(self, n: int):
        res = []

        def dfs(str, left, right):

            if left == 0 and right == 0:
                res.append(str)

            if right < left:
                return

            if left > 0:
                dfs(str+'(', left-1, right)
            if right > 0:
                dfs(str+')', left, right-1)
        dfs("", n, n)
        return res


if __name__ == "__main__":
    n = 3
    result = Solution().generateParenthesis(3)
    print(result)
