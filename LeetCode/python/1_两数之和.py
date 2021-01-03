class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = {}
        for index, val in enumerate(nums):
            if target - nums[index] in hashtable:
                return [hashtable[target - nums[index]], index]
            hashtable[nums[index]] = index

        return [0, 0]