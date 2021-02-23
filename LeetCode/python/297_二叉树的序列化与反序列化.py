

from collections import deque


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return []

        res = []
        q = deque()
        q.append(root)
        while q:
            node = q.popleft()

            if node is None:
                res.append('null')
            else:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)

        return res

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data or len(data) == 0:
            return None
        self.root = TreeNode(data[0])
        queue = deque([self.root])
        leng = len(data)
        nums = 1
        while nums < leng:
            node = queue.popleft()
            if node:
                node.left = TreeNode(data[nums]) if data[nums] else None
                queue.append(node.left)
                if nums + 1 < leng:
                    node.right = TreeNode(data[nums + 1]) if data[nums + 1] else None
                    queue.append(node.right)
                    nums += 1
                nums += 1
        return self.root



