class Solution:
    def singleNumber(self, nums):

        s = 0
        for i in nums:
            s ^= i

        return s

