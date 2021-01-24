# Definition for singly-linked list.

"""
    思路1：对于所有的链表，两两合并
    
"""

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def mergeKLists(self, lists):
        import heapq
        dummy = ListNode(-1)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(head, [lists[i].val, i])
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            node = ListNode(val)
            p.next = node
            p = p.next
            if lists[idx]:
                heapq.heappush(head, [lists[idx].val, idx])
                lists[idx] = lists[idx].next

        return dummy.next