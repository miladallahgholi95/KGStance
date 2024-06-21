import random
import networkx as nx


class MultiSourceRandomWalk:
    def __init__(self, data):
        self.graph = nx.Graph()
        self.build_graph(data)

    def build_graph(self, data):
        for word1, word2, type_ in data:
            self.graph.add_edge(word1, word2, type=type_)

    def random_walk(self, sources, walk_length, num_walks):
        node_visits = {node: 0 for node in self.graph.nodes()}
        for source in sources:
            for _ in range(num_walks):
                current_node = source
                for _ in range(walk_length):
                    neighbors = list(self.graph.neighbors(current_node))
                    if not neighbors:
                        break
                    current_node = random.choice(neighbors)
                    node_visits[current_node] += 1

        return node_visits

    def get_important_nodes(self, sources, walk_length, num_walks):
        visits = self.random_walk(sources, walk_length, num_walks)
        important_nodes = sorted(visits.items(), key=lambda item: item[1], reverse=True)
        return important_nodes

