class Solution:
    """
    思路一：二分查找，设立上下标
    思路二：顺序查找
    """
    def searchRange(self, nums, target):
        m = -1
        n = len(nums)
        for i in range(len(nums)):
            if nums[i] == target:
                if i < n:
                    n = i
                if i >= m:
                    m = i

        if m == -1:
            n = -1

        return [n, m]


if __name__ == "__main__":
    nums = [5, 7, 7, 8, 8, 10]
    target = 6
    result = Solution().searchRange(nums, target)