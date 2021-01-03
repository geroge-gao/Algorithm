from input_utils import ListNode, create_list_node, print_list


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
            flag = int((flag + l1.val + l2.val)/10)
            q = ListNode(val)
            p.next = q
            p = q
            l1 = l1.next
            l2 = l2.next

        while l1 is not None:
            val = (l1.val + flag) % 10
            flag = int((l1.val + flag) / 10)
            p.next = ListNode(val)
            p = p.next
            l1 = l1.next

        while l2 is not None:
            val = (l2.val + flag) % 10
            flag = int((l2.val + flag) / 10)
            p.next = ListNode(val)
            p = p.next
            l2 = l2.next

        if flag != 0:
            p.next = ListNode(flag)

        return start_node


if __name__ == "__main__":
    l1 = [9,9,9,9,9,9,9]
    l2 = [9,9,9,9]
    l1 = create_list_node(l1)
    l2 = create_list_node(l2)
    result = Solution().addTwoNumbers(l1, l2)

    print_list(result)




