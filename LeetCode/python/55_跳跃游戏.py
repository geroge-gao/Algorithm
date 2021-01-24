class Solution:
    """
    关键点，index+ value
    计算每个位置能够到达的最远位置
    """

    def canJump(self, nums):
        count = len(nums)
        flag = [False] * count
        flag[0] = True
        i = 0
        right_most = 0
        while i < count:
            if i <= right_most:
                right_most = max(right_most, i+nums[i])
                if right_most >= count - 1:
                    return True
            i += 1

        return False

if __name__ == "__main__":
    data = [0,2,3]
    result = Solution().canJump(data)



