from input_utils import Tree

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    """
    二叉树构建：
    先序遍历，创建二叉树
    """

    def create_tree(self, t1,t2):
        if not t1 and not t2:
            return None
        root = TreeNode(0)
        if t1 and t2:
            root.val = t1.val + t2.val
            root.left = self.create_tree(t1.left, t2.left)
            root.right = self.create_tree(t1.right, t2.right)
        elif t1:
            root.val = t1.val
            root.left = self.create_tree(t1.left, None)
            root.right = self.create_tree(t1.right, None)
        elif t2:
            root.val = t2.val
            root.left = self.create_tree(None, t2.left)
            root.right = self.create_tree(None, t2.right)

        return root



    def mergeTrees(self, t1, t2):

        return self.create_tree(t1, t2)

if __name__ == "__main__":
    l1 = [1,3,2,5]
    l2 = [2,1,3,None,4,None,7]
    t1 = Tree()
    head1 = t1.construct_tree(l1)
    t2 = Tree()
    head2 = t2.construct_tree(l2)
    result = Solution().mergeTrees(head1, head2)

