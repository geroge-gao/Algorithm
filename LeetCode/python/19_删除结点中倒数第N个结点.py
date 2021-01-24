def get_length(head):
    # 获取链表长度
    p = head
    count = 0
    while p is not None:
        count += 1
        p = p.next

    return count


class Solution:
    def removeNthFromEnd(self, head, n):
        m = get_length(head)

        # 返回头结点
        if m == n:
            return head.next
        k = m - n - 1
        p = head
        for i in range(k):
            p = p.next

        # 删除第n个结点
        q = p.next
        if q is not None:
            p.next = q.next
        else:
            p.next = q

        return head