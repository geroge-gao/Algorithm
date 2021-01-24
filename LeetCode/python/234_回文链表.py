# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):
        p = head
        a = []
        while p:
            a.append(p.val)
            p = p.next

        if len(a) > 0:
            i = 0
            j = len(a) - 1
            while i < j:
                if a[i] == a[j]:
                    i += 1
                    j -= 1
                else:
                    return False

        return True