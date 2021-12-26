class Solution:

    def combinationSum(self, candidates, target):

        res_all = []

        def dfs(self, candiates, res, target):
            nonlocal res_all
            if sum(res) == target:
                self.res_all.append(res)
                return
            elif sum(res) > target:
                return

            for i in range(len(candiates)):
                self.dfs(candiates[i:], res + [candiates[i]], target)

        dfs(candidates,[], target)

        return res_all