class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def create_list_node(data):
    start_node = ListNode(data[0])
    p = start_node
    for i in range(1, len(data)):
        q = ListNode(data[i])
        p.next = q
        p = q
    return start_node


def print_list(list_node):
    p = list_node
    while p is not None:
        print(p.val, '')
        p = p.next
