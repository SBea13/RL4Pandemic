import numpy as np
from Graph import Graph
from Graph import SimpleNode
import configparser
import networkx as nx
import scipy as sp


_graph_types = []

def graph_type(cls):
    _graph_types.append(cls)
    return cls

def make_graph(type):
    for gt in _graph_types:
        if gt._name == type:
            return gt
    raise NotImplementedError('Graphs of type "{}" are not implemented'.format(type))
    

@graph_type
class ErdosRenyiGraph(Graph):
    """Constructs a Erdos Renyi Graph (aka binomial graph). @N nodes (each at a random position x, y in [-@L,@L]) are connected by edges, so that any given pair of nodes is connected with a probability @p
    """
    
    _name = "ErdosRenyi"
    
    def __init__(self, N, p, seed=None, nodetype=SimpleNode, L=5):
        """Generate an Erdos Renyi Graph

        Parameters
        ----------
        N : int
            Number of nodes
        p : float
            Probability that each pair of nodes is connected by an edge (with random weight)
        seed : int, optional
            seed for random generator, by default None
        nodetype : Node (class), optional
            type of nodes in the graph, by default SimpleNode
        L : float, optional
            Nodes positions are chosen uniformly inside the square [-L,L]x[-L,L], by default 5
        """
        
        positions = np.random.uniform(-L, L, size=(N,2))
        graph = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
        
        adj = nx.adjacency_matrix(graph).todense()
        
        weights = np.random.uniform(0, 1, size=(N,N))
        weights = np.tril(weights) + np.tril(weights, -1).T
        
        adj = np.multiply(adj, weights)
        
        super().__init__(adj, positions, node_type=nodetype)

@graph_type
class KarateClub(Graph):
    _name = "KarateClub"
    
    def __init__(self, L=5, nodetype=SimpleNode, **kwargs):
        G = nx.karate_club_graph()
        positions = np.random.uniform(-L, L, size=(len(G),2))
        adj = nx.adjacency_matrix(G).todense()
        
        super().__init__(adj, positions, node_type=nodetype)
    
@graph_type
class GnmGraph(Graph):
    _name = "Gnm"
    
    def __init__(self, N, m, seed=None, nodetype=SimpleNode, L=5.):
        
        positions = np.random.uniform(-L, L, size=(N,2))
        graph = nx.dense_gnm_random_graph(N, m, seed=seed)
        adj = nx.adjacency_matrix(graph).todense()
        
        super().__init__(adj, positions, node_type=nodetype)

@graph_type
class PreferentialAttachment(Graph):
    """
    Grow a graph from a small Erdos-Renyii Graph by adding nodes, each connected to other nodes with a probability proportional to the target's degree. So, nodes with many connections tend to "receive" more connections (Barabasi-Albert model).
    """
    
    _name = "PreferentialAttachment"
    
    def __init__(self, N, p, nstart=10, m=3, seed=None, nodetype=SimpleNode, L=5):
        """[summary]

        Parameters
        ----------
        N : [type]
            [description]
        p : [type]
            [description]
        nstart : int, optional
            [description], by default 10
        m : int, optional
            minimum number of connections for new nodes, by default 3
        seed : [type], optional
            [description], by default None
        nodetype : [type], optional
            [description], by default SimpleNode
        L : int, optional
            [description], by default 5

        Raises
        ------
        ValueError
            [description]
        """
        
        if (N <= nstart):
            raise ValueError("Graph too small")
            
        positions = np.random.uniform(-L, L, size=(N,2))
        
        graph = nx.erdos_renyi_graph(nstart, p, seed=seed, directed=False)
        adj = nx.adjacency_matrix(graph).todense()
        
        weights = np.random.uniform(0, 1, size=(nstart,nstart))
        weights = np.tril(weights) + np.tril(weights, -1).T
        
        adj = np.multiply(adj, weights)
        
        super().__init__(adj, positions[:nstart], node_type=nodetype)
        
        for i in range(nstart, N-nstart):
            #Compute probabilities of connections
            id_probas = np.array([[node.id, node.degree] for node in list(self.vertList.values())])
            
            probas = id_probas[:,1] / np.sum(id_probas[:,1])
            cum_probas = np.cumsum(probas)
            
            toNodes = np.random.choice(id_probas[:,0], size=m, replace=False, p=probas)            
               
            fromNode = self.addVertex(i, positions[i])
            
            weights = np.random.uniform(0, 1, size=m)
            
            print(probas)
            print(toNodes)
            for j, to in enumerate(toNodes):
                print(fromNode)
                self.addEdge(fromNode.id, to, weight=weights[j])
                self.addEdge(to, fromNode.id, weight=weights[j])
                print(self.getVertex(to))


    
config = configparser.ConfigParser()
config_rtn = config.read('parameters.txt')
config = config["DEFAULT"]

if config_rtn == []:
    raise ValueError("Missing configuration")

if __name__ == "__main__":

    gen = make_graph("PreferentialAttachment")
    g = gen(30, p=.5, seed = 42)
    g.plot(nodeSize=.5)

            
    input("Press Enter to exit...")
    
    
        
    