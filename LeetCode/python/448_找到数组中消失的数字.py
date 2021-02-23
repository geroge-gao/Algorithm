class Solution:
    def findDisappearedNumbers(self, nums):

        result = []
        n = len(nums)
        data = {}
        for i in range(n):
            if nums[i] not in data:
                data[nums[i]] = 1

        for i in range(1, n+1):
            if i not in data:
                result.append(i)

        return result
