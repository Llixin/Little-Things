import math
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx


class CommunityDetection:
    def __init__(self, path):
        self.graph = None
        self.get_graph(path)

    def get_graph(self, path):
        with open(path) as f:
            text_lines = f.readlines()
        graph = nx.Graph()
        for line in text_lines:
            line = line.replace('\n', '')
            if not line:
                continue
            u, v = list(map(int, line.split(' ')))
            graph.add_edge(u, v)
        self.graph = graph

    def get_status_1(self, i):
        res = 0
        for j in self.graph.adj[i]:
            if self.graph.nodes[j].get("status", 0) == 1:
                res += 1
        return res

    def cal_similarity(self, i, j):
        TE = lambda x: self.get_status_1(x) / self.graph.degree(x)
        TS = lambda x: sum(TE(y) for y in self.graph.adj[x]) / abs(sum(1 for _ in self.graph.adj[x]))
        TCS = lambda x, y: TS(x) * TS(y) / math.sqrt(self.graph.degree(x) * self.graph.degree(y))
        return TCS(i, j)

    def cal_modularity(self, community):
        def delta(x, y):
            for com in community:
                if x in com:
                    return 1 if y in com else 0
            return 0

        M = self.graph.number_of_edges()
        is_adj = lambda x, y: 1 if x in self.graph.adj[y] else 0

        Q = 0
        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if i == j:
                    continue
                Q += (is_adj(i, j) - self.graph.degree(i) * self.graph.degree(j) / (2 * M)) * delta(i, j)
        Q /= 2 * M
        return Q

    def detect(self):
        node = list(self.graph.nodes())
        c, q = [], float('-inf')
        i, n = 0, len(node)
        self.graph.nodes[node[i]]['status'] = 1
        i = (i + 1) % n
        while self.graph.number_of_edges() >= self.graph.number_of_nodes():
            self.graph.nodes[node[i]]['status'] = 1
            i = (i + 1) % n
            u, v, mTCS = 0, 0, float('inf')
            for a, b in self.graph.edges():
                TCS = self.cal_similarity(a, b)
                if TCS < mTCS:
                    u, v, mTCS = a, b, TCS
            self.graph.remove_edge(u, v)
            community = [list(subgraph) for subgraph in nx.connected_components(self.graph)]
            Q = self.cal_modularity(community)
            if q < Q:
                c, q = community, Q
            print('delete: {}--{}, TCS:{}, Q:{}'.format(u, v, mTCS, Q))
        one = []
        i = 0
        while i < len(c):
            if len(c[i]) == 1:
                one.extend(c.pop(i))
            else:
                i += 1
        for u, v in list(combinations(one, 2)):
            self.graph.add_edge(u, v)
        c.append(one)
        q = self.cal_modularity(c)
        return c, q


def get_color(G, community):
    val_map = defaultdict(int)
    for i, com in enumerate(community):
        for q in com:
            val_map[q] = i
    values = [val_map.get(node, "red") for node in G.nodes()]
    return values


def main():
    path = "data/club.txt"
    test = CommunityDetection(path)
    community, Q = test.detect()
    print('community:', community)
    print('Q:', Q)
    g = test.graph
    values = get_color(g, community)
    nx.draw(g, node_color=values, with_labels=True)
    plt.show()


if __name__ == "__main__":
    main()
