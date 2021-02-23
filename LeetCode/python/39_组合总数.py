class Solution:
    def __init__(self):
        self.res_all = []

    def dfs(self, candiates, res, target):

        if sum(res) == target:
            self.res_all.append(res)
            return
        elif sum(res) > target:
            return

        for i in range(len(candiates)):
            self.dfs(candiates[i:], res + [candiates[i]], target)

    def combinationSum(self, candidates, target):

        self.dfs(candidates,[], target)

        return self.res_all