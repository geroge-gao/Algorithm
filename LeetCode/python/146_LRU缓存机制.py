import collections

class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache(collections.OrderedDict):
    """
    思路一：采用OrderedDict，面试一般需要自己实现
    思路二：哈希表+双向链表
    """

    def __init__(self, capacity: int):
        self.cache = {}
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0


    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:

        if key not in self.cache:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            if self.size >= self.capacity:
                removed = self.removeTail()
                self.cache.pop(removed.key)
            self.addToHead(node)
        else:
            # 如果存在哈希冲突，修改结果并将其移动到链表头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
            print("哈希冲突")


    def addToHead(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

# Your LRUCache object will be instantiated and called as such:


if __name__ == '__main__':

    lRUCache = LRUCache(2)
    lRUCache.put(1, 0)
    lRUCache.put(2, 2)

    print(lRUCache.get(1))

    lRUCache.put(3, 3)

    print(lRUCache.get(2))
    lRUCache.put(4, 4)


    print(lRUCache.get(1))
    print(lRUCache.get(3))

    print(lRUCache.get(4))

