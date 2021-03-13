class Solution:
    """
    贪心+排序: 首先将身高按照降序排序，身高高的优先放到前面
    个子矮的对于个子高的没有影响，因此个子矮的一般要放到后面

    """
    def reconstructQueue(self, people):
        people.sort(key=lambda x: (-x[0], x[1]))
        output = []
        for p in people:
            output.insert(p[1], p)

        return output

if __name__ == '__main__':
    people = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
    res = Solution().reconstructQueue(people)
    print(res)
