class Solution:
    """和309一个类型的题目
    判断每个状态偷还是不偷
    """
    def rob(self, nums):

        n = len(nums)

        if n == 0 or n == 1:
            return sum(nums)
        f = [[0] * 2 for _ in range(n)]
        f[0][0] = 0  # 不偷
        f[0][1] = nums[0]  # 偷
        f[1][0] = max(f[0][0], f[0][1])
        f[1][1] = max(f[0][0]+nums[1], f[0][1])

        for i in range(1, n):
            f[i][0] = max(f[i-1][1], f[i-1][0])
            f[i][1] = f[i-1][0] + nums[i]

        return max(f[n-1][0], f[n-1][1])


if __name__ == '__main__':
    nums = [1, 2, 3, 1]
    res = Solution().rob(nums)
    print(res)



