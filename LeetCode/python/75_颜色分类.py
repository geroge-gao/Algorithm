class Solution:
    """
    思路一：sort函数
    思路二：单指针，第一次遍历，放0的位置，第二次遍历，放一的位置
    思路二：双指针，如果是0异动到前面，2移动到后面
    """
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        p, q = 0, length - 1
        i = 0
        while i <= q:
            if nums[i] == 0:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1

            if nums[i] == 2:
                nums[q], nums[i] = nums[i], nums[q]
                q -= 1

            i += 1
