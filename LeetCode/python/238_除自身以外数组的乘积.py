class Solution:
    def productExceptSelf(self, nums):

        n = len(nums)
        L, R, res = [0] * n, [0] * n, [0] * n
        L[0], R[n-1] = 1, 1
        for i in range(1, n):
            L[i] = L[i-1] * nums[i-1]

        for i in reversed(range(n-1)):
            R[i] = R[i+1] * nums[i+1]

        for i in range(n):
            res[i] = L[i] * R[i]

        return res
