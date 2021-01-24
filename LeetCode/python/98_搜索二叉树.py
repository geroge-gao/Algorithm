class Solution:

    """
    常规题： 搜索二叉树的中序遍历为有序递增的
    """

    def __init__(self):
        self.result = []

    def preorder(self, root):
        # 先序遍历
        if root:
            self.preorder(root.left)
            self.result.append(root.val)
            self.preorder(root.right)

    def check_monotonicity(self, data):
        nums = len(data)
        for i in range(nums-1):
            j = i+1
            if data[i] < data[j]:
                continue
            else:
                return False

        return True

    def isValidBST(self, root):

        self.preorder(root)
        return self.check_monotonicity(self.result)