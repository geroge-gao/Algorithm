# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from input_utils import create_list_node

class Solution:
    def detectCycle(self, head):

        if not head:
            return None

        point_slow = head
        point_fast = head
        count = 0

        while point_slow and point_fast:

            if point_slow == point_fast and count != 0:
                point_fast = head
                break

            point_slow = point_slow.next

            point_fast = point_fast.next
            if point_fast:
                point_fast = point_fast.next

            count += 1

        # 如果不存在环
        if not point_slow or not point_fast:
            return None

        # index = 0
        while point_fast != point_slow:
            point_fast = point_fast.next
            point_slow = point_slow.next
        return point_fast

if __name__ == "__main__":
    data = [1, 2]
    head = create_list_node(data)
    p = head
    while p.next:
        p = p.next
    p.next = head

    result = Solution().detectCycle(head)
    print(result)


