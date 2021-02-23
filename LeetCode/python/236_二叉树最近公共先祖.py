class Solution:

    """
    求最近公共祖先，将右节点指向父节点，就能转换成链表求父节点的问题
    """
    def length(self, node):
        count = 0
        while node:
            count += 1
            node = node.right

        return count

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        def inorder(node, parent):

            if node:
                inorder(node.left, node)
                inorder(node.right, node)
                node.right = parent

        inorder(root, None)

        l1 = self.length(p)
        l2 = self.length(q)

        if l1 > l2:
            i = 0
            while i < l1-l2:
                p = p.right
                i += 1
        else:
            i = 0
            while i < l2 - l1:
                q = q.right
                i += 1

        while p and q:
            if p == q:
                return p
            p = p.right
            q = q.right
