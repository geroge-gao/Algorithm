class Solution:
    """
    思路一：比较简单，动态规划， O(n^2),从后往前遍历
    思路二：
    """
    def lengthOfLIS(self, nums):

        length = len(nums)

        dis = [1] * length
        dis[0] = 1
        max_len = 1

        for i in range(1, length):
            j = i - 1
            while j >= 0:
                if nums[i] > nums[j]:
                    dis[i] = max(dis[j]+1, dis[i])
                j -= 1
            if dis[i] > max_len:
                max_len = dis[i]

        return max_len
