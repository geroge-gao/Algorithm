
class UnionFind:
    # 定义一个并查集类
    # 特点：一边查询，一边修改节点指向

    def __init__(self):
        self.parent = {}
        self.weight = {}

    def find(self, x):
        # 路径压缩
        # 找到根节点

        root = x
        multi = 1

        # if x in self.parent:
        # 判断查询语句是否有效
        while self.parent[root] != root:
            # 计算路径上的权值
            multi *= self.weight[root]
            root = self.parent[root]
        while x != root:
            last_parent = self.parent[x]
            cur_weight = self.weight[x]
            self.weight[x] = multi
            multi /= cur_weight
            self.parent[x] = root
            x = last_parent

        return root

    def merge(self, x, y, val):
        # 合并并查集
        parent_x = self.find(x)
        parent_y = self.find(y)

        if parent_x == parent_y:
            return

        if parent_x != parent_y:
            self.parent[parent_x] = parent_y
            # 合并之后更新权值
            self.weight[parent_x] = self.weight[y] * val/self.weight[x]

    def is_connected(self, x, y):
        # 判断两点是否相连
        return x in self.parent and y in self.parent and self.find(x) == self.find(y)

    def add(self, x):

        if x not in self.parent:
            self.parent[x] = x
            self.weight[x] = 1.0


class Solution:
    """
    思路：带权重并查集

    """
    def calcEquation(self, equations, values, queries):
        uf = UnionFind()
        # 构建并查集
        for (a, b), val in zip(equations, values):
            uf.add(a)
            uf.add(b)
            uf.merge(a, b, val)

        res = []

        for (a, b) in queries:
            if uf.is_connected(a, b):
                res.append(uf.weight[a]/uf.weight[b])
            else:
                res.append(-1.0)

        return res


if __name__ == '__main__':
    equations = [["a","b"],["e","f"],["b","e"]]
    values = [3.4,1.4,2.3]
    queries = [["b","a"],["a","f"],["f","f"],["e","e"],["c","c"],["a","c"],["f","e"]]
    res = Solution().calcEquation(equations, values, queries)
    print(res)


