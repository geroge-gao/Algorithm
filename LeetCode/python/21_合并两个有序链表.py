class Solution:
    def mergeTwoLists(self, l1, l2):
        head = ListNode(0, None)
        node = head
        p = l1
        q = l2
        while p is not None and q is not None:
            if p.val <= q.val:
                node.next = p
                node = p
                p = p.next
            else:
                node.next = q
                node = q
                q = q.next

        if p is not None:
            node.next = p
        if q is not None:
            node.next = q

        return head.next

