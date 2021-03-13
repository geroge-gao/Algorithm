class Solution:
    """
    思路：dfs+剪枝
    """
    def findTargetSumWays(self, nums, S):

        count = 0
        n = len(nums)

        def dfs(nums, start, target, S):
            nonlocal count, n
            if start == n:
                if target == S:
                    count += 1
            else:
                if start < len(nums):
                    dfs(nums, start+1, target+nums[start], S)
                    dfs(nums, start+1, target-nums[start], S)

        if S > sum(nums) or S < -sum(nums):
            return count
        else:
            dfs(nums, 0, 0, S)

        return count

if __name__ == '__main__':
    nums = [38,21,23,36,1,36,18,3,37,27,29,29,35,47,16,0,2,42,46,6]
    S = 14
    res = Solution().findTargetSumWays(nums, S)
    print(res)
