class Solution:
    def permute(self, nums):
        m = len(nums)
        visited = [False] * (m+1)
        res = []

        def dfs(linklist, n):
            if n == 0:
                res.append(linklist)

            for i in range(1, m+1):
                if not visited[i]:
                    visited[i] = True
                    dfs(linklist + [nums[i-1]], n - 1)
                    visited[i] = False

        dfs([], m)
        return res