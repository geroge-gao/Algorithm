from collections import deque
class Solution:
    """
    思路一：暴力求解，直接找出方框里面最大的值，基本超时
    思路二：双端队列，其实就是利用双端队列实现最大堆，每次在结果里面保存可能最大的值
    """
    def maxSlidingWindow(self, nums, k):

        n = len(nums)

        q = deque()
        res = []
        for i in range(k):
            if not q or nums[i] <= nums[q[-1]]:
                q.append(i)
            else:
                while q and nums[i] > nums[q[-1]]:
                    q.pop()
                q.append(i)

        res.append(nums[q[0]])
        for i in range(k, n):
            while q and nums[i] > nums[q[-1]]:
                q.pop()
            q.append(i)
            if i - q[0] == k:
                q.popleft()

            res.append(nums[q[0]])
        return res
