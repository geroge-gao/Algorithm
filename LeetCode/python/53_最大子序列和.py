class Solution:
    def maxSubArray(self, nums):

        cur = 0
        sum_all = -float('inf')
        for i in range(len(nums)):

            if cur > 0:
                cur += nums[i]
            else:
                cur = nums[i]

            if cur > sum_all:
                sum_all = cur

        return sum_all