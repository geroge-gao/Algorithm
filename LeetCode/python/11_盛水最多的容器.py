class Solution:
    def maxArea(self, height) -> int:

        n = len(height)
        max_area = 0
        l, r = 0, n-1

        while l < r:
            area = min(height[l], height[r]) * (r-l)
            max_area = max(area, max_area)

            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1

        return max_area