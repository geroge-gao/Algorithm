class Solution:
    """
    思路一：将
    """
    def moveZeroes(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        index = 0
        count = len(nums)
        for i in range(count):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1
        while index < count:
            nums[index] = 0
            index += 1

        return nums


