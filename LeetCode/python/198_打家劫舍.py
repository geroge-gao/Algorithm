class Solution:
    """和309一个类型的题目
    判断每个状态偷还是不偷
    """
    def rob(self, nums):

        n = len(nums)

        if n == 0 or n == 1:
            return sum(nums)
        f = [0] * n
        f[0] = nums[0]
        f[1] = max(nums[0], nums[1])

        for i in range(2, n):
            f[i] = max(nums[i]+f[i-2], f[i-1])

        return f[n-1]