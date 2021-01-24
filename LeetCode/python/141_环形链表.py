# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

from input_utils import create_list_node


class Solution:
    # 基础题，以前做过，快慢指针
    def hasCycle(self, head):

        if not head or not head.next:
            return False

        low = head
        high = head.next

        flag = False

        while low and high:

            if low == high:
                return True
            low = low.next
            high = high.next
            if high:
                high = high.next

        return False

if __name__ == "__main__":
    data = [1, 2]
    head = create_list_node(data)
    # p = head
    # while p.next:
    #     p = p.next
    # p.next = head.next

    result = Solution().hasCycle(head)
    print(result)

