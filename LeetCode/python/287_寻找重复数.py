class Solution:
    def findDuplicate(self, nums):
        flag = dict(zip(nums, [0]* len(nums)))
        for i in nums:
            if flag[i] == 0:
                flag[i] += 1
            else:
                return i
