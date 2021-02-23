class Solution:
    """
    第一反应是回溯
    """
    def subsets(self, nums):

        count = len(nums)
        res_all = []

        def dfs(data, target, res):

            if target == 0:
                res_all.append(res.copy())
                return

            for i in range(len(data)):
                res.append(data[i])
                dfs(data[i+1:], target-1, res)
                res.pop()

        res_all.append([])

        for i in range(1, count+1):
            dfs(nums, i, [])

        return res_all