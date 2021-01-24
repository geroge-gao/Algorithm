class Solution:
    def threeSum(self, nums):

        nums.sort()
        count = len(nums)
        result = []
        first = -1
        second = -1
        third = -1
        for i in range(count-1):
            j = i + 1
            k = count - 1

            while j < k:
                target = nums[i] + nums[j] + nums[k]
                if target < 0:
                    j += 1
                elif target > 0:
                    k -= 1
                else:
                    if third == -1 or (nums[j] > nums[second] or nums[k] < nums[third] or nums[i] > nums[first]):
                        # 第一次满足条件或者不存在重复的时候
                        result.append([nums[i], nums[j], nums[k]])
                        # 更新三个元祖的游标
                        first = i
                        second = j
                        third = k

                    j += 1

        return result


if __name__ == "__main__":
    nums = [-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6]
    result = Solution().threeSum(nums)
    print(result)


