# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from input_utils import Tree

class Solution:

    """
    最大连续序列和问题类似：没有考虑最大值为负值情况 ...似乎有点问题
    改变思路：
    都要经过父节点，那么只需要算子节点的最大值就行了
    父节点等于子节点求和
    """

    def __init__(self):

        self.max_distance = -float('inf')

    def dfs(self, root):
        if not root:
            return 0

        left = max(self.dfs(root.left), 0)
        right = max(self.dfs(root.right), 0)

        cur = root.val + left + right
        if cur > self.max_distance:
            self.max_distance = cur


        # 返回的结果是二叉树深度中的一点
        return root.val + max(left, right)


    def maxPathSum(self, root):
        self.dfs(root)
        return self.max_distance

if __name__ == "__main__":
    t = Tree()
    data = [5,4,8,11,None,13,4,7,2,None,None,None,1]
    root = t.construct_tree(data)
    result = Solution().maxPathSum(root)
    print(result)


