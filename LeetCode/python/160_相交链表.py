# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    """
    常规题，先算两个链表的长度，m, n
    然后长的链表先移动m-n步，然后同时移动链表，
    当两个链表节点相等时，表示该节点是相交节点。
    """

    def get_length(self, head):
        length = 0
        while head:
            length += 1
            head = head.next

        return length


    def getIntersectionNode(self, headA, headB):
        p = headA
        q = headB

        m = self.get_length(p)
        n = self.get_length(q)

        diff = m - n
        if diff >= 0:
            for i in range(abs(diff)):
                p = p.next
        else:
            for i in range(abs(diff)):
                q = q.next

        while p != q:
            p = p.next
            q = q.next

        return p
