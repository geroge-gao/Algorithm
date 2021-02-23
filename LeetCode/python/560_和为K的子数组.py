class Solution:
    """
    思路一：暴力枚举，不解释连招，超时
    思路二：利用哈希记录前缀值
    """

    def subarraySum(self, nums, k):
        pre = {0:1}
        cur = 0
        count = 0
        for i in range(len(nums)):
            cur += nums[i]
            if cur - k in pre:
                count += pre[cur - k]

            if cur in pre:
                pre[cur] += 1
            else:
                pre[cur] = 1

        return count

