class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        val = [1] + nums + [1]

        @lru_cache(None)
        def solve(left: int, right: int) -> int:
            if left >= right - 1:
                return 0

            best = 0
            for i in range(left + 1, right):
                total = val[left] * val[i] * val[right]
                total += solve(left, i) + solve(i, right)
                best = max(best, total)

            return best

        return solve(0, n + 1)