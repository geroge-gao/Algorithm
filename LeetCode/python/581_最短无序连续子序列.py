class Solution:
    def findUnsortedSubarray(self, nums):
        target = sorted(nums)

        start = len(nums)
        end = 0
        max_len = 0
        for i in range(len(nums)):

            if nums[i] != target[i]:
                start = min(i, start)
                end = max(i, end)

            if start > end:
                max_len = 0
            else:
                max_len = end - start + 1

        return max_len

if __name__ == "__main__":
    data = [1, 2, 3, 4]
    result = Solution().findUnsortedSubarray(data)
    print(result)



