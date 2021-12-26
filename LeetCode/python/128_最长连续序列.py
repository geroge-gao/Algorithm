#
class Solution:
    """
    思路一：排序 + 双指针
    """
    def longestConsecutive(self, nums):

        nums = sorted(nums)
        length = len(nums)

        if length < 2:
            return length

        start = 0
        end = 0
        cur = 1
        max_len = 0
        while cur < length:
            if nums[end] + 1 == nums[cur] or nums[end] == nums[cur]:
                end = cur
            else:
                end = cur
                start = end

            l = len(set(nums[start:end+1]))

            if l > max_len:
                max_len = l

            cur += 1

        return max_len

if __name__ == '__main__':
    data = [1,1,1,1,2,2,3,2,5]
    res = Solution().longestConsecutive(data)
    print(res)

