# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def buildTree(self, preorder, inorder) -> TreeNode:

        if not preorder and not inorder:
            return None

        root_node = preorder[0]
        root = TreeNode(root_node)
        root_index = inorder.index(root_node)

        # 如果只有一个节点，直接返回
        if len(preorder) == 1:
            return root

        in_left = inorder[:root_index]
        in_right = inorder[root_index+1:]

        pre_left = preorder[1:root_index+1]
        pre_right = preorder[root_index+1:]

        root.left = self.buildTree(pre_left, in_left)
        root.right = self.buildTree(pre_right, in_right)

        return root
