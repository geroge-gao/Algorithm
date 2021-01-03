
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        flag = int((l1.val + l2.val)/10)
        val = (l1.val+l2.val) % 10
        start_node = ListNode(val)
        p = start_node
        l1 = l1.next
        l2 = l2.next
        while l1 is not None and l2 is not None:
            val = (flag + l1.val + l2.val)%10
            flag = (flag + l1.val + l2.val)/10
            q = ListNode(val)
            p.next = q
            p = q
            l1 = l1.next
            l2 = l2.next

        if l1 is not None:
            p.next = l1
        if l2 is not None:
            p.next = l2

        return start_node
