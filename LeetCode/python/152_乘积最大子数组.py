class Solution:
    """
    两个数组，最大值和最小值
    """
    def maxProduct(self, nums):
        n = len(nums)
        max_res = [1] * (n+1)
        min_res = [1] * (n+1)

        for i in range(1, n+1):
            max_res[i] = max(max_res[i-1] * nums[i-1], min_res[i-1] * nums[i-1], nums[i-1])
            min_res[i] = min(max_res[i-1] * nums[i-1], min_res[i-1] * nums[i-1], nums[i-1])

        return max(max_res[1:])


if __name__ == '__main__':
    data = [2, 3, -2, 4]
    res = Solution().maxProduct(data)
    print(res)
