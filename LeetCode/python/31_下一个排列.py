
class Solution:
    def nextPermutation(self, nums) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        i = len(nums) - 2
        while i >= 0:
            if nums[i] < nums[i+1]:
                break
            i -= 1

        if i < 0:
            nums.reverse()
        else:
            j = len(nums) - 1
            while j > i:
                if nums[j] > nums[i]:
                    nums[j], nums[i] = nums[i], nums[j]
                    break
                j -= 1
            left = i + 1
            right = len(nums) - 1
            # 交换之后，i+1-n为降序，换成升序
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1

