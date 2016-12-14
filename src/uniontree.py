class UnionTree:
    def __init__(self, datas):
        n = len(datas)
        keys = range(n)
        nodes = [Node(key, data=data) for (key, data) in zip(keys, datas)]
        self.nodes = dict(zip(keys, nodes))
        self.top = list(keys)
        self.nextkey = n

    def merge(self, data, childkeys):
        for key in childkeys:
            self.top.remove(key)
        rootkey = self.nextkey
        self.nextkey += 1
        self.nodes[rootkey] = Node(rootkey, data=data, children=childkeys)
        self.top.append(rootkey)

    def node(self, key):
        return self.nodes[key]

    def roots(self):
        return [self.nodes[key] for key in self.top]

    def children(self, node):
        return [self.nodes[key] for key in node.children]

    def nodestr(self, node):
        return node.__str__() + ", children=["\
            + "; ".join([self.nodestr(childnode)\
            for childnode in self.children(node)]) + "]"

    def __str__(self):
        return "; ".join([self.nodestr(node) for node in self.roots()])

    def __repr__(self):
        return self.__str__()

class Node:
    def __init__(self, key, data=None, children=[]):
        self.key = key
        self.data = data
        self.children = children

    def __str__(self):
        return "data=" + self.data.__str__()

    def __repr__(self):
        return self.__str__()
