class Solution:
    """
    分桶的思想：
    """
    def leastInterval(self, tasks, n) -> int:
        import collections
        freq = collections.Counter(tasks)

        # 最多的执行次数
        max_exec = max(freq.values())
        # 具有最多执行次数的任务数量
        max_count = sum(1 for v in freq.values() if v == max_exec)
        # 总排队时间 = (桶个数 - 1) * (n + 1) + 最后一桶的任务数
        return max((max_exec - 1) * (n + 1) + max_count, len(tasks))


if __name__ == '__main__':
    tasks = ["A", "A", "A", "B", "B", "B"]
    n = 2
    res = Solution().leastInterval(tasks, n)
    print(res)

