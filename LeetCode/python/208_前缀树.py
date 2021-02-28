class TrieNode:

    def __init__(self, val=None, children=[], end=False):
        self.val = val
        self.children = children
        self.end = end

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 创建一个根节点
        self.node = {}



    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.node
        for w in word:
            if w not in node:
                node[w] = {}
            node = node[w]


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.node

        for w in word:
            if w not in node:
                return False
            node = node[w]

        return True

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.node
        for p in prefix:
            if p not in node:
                return False
            node = node[p]

        return True



# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)