# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from input_utils import create_list_node

class Solution:

    """
    常规题: 考虑边界条件
    """
    def reverseList(self, head):

        if not head or not head.next:
            return head

        p = head
        q = head.next
        p.next = None
        while q.next:
            r = q.next
            q.next = p
            p = q
            q = r

        q.next = p
        return q


if __name__ == "__main__":

    data = [1, 2, 3, 4, 5]
    head = create_list_node(data)
    result = Solution().reverseList(head)

