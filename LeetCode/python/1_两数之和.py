class Solution:
    def twoSum(self, nums, target):
        hashtable = {}
        for index, val in enumerate(nums):
            if target - nums[index] in hashtable:
                return [hashtable[target - nums[index]], index]
            hashtable[val] = index
        return [0, 0]


if __name__ == '__main__':
    nums = [3, 2, 4]
    target = 6
    res = Solution().twoSum(nums, target)
    print(res)