import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

class SimpleNode:    
    """Basic Node in a Graph"""
    
    def __init__(self, identifier, position):
        """Create a node named @identifier, located at 2D @position (only matters for plotting)

        Parameters
        ----------
        identifier : hashable
            Name of the node
        position : ndarray of shape (2,)
            2D position of the node, used for plotting
        """
        
        self.id = identifier
        self.position = position
        
        self.connectedTo = {} #Dictionary of pairs (instance of Node, weight of connection)
        
        #Node local information
        self.degree = 0
        self.total_weight = 0
        
        #Plotting parameters
        self.color = "#FFFFFF"  #Color (default is white)
        self.isolated = False   #If True, draw a border around the node
    
    def addNeighbour(self, to, weight=0):
        """Add a single connection from @self to the Node @to, weighted by @weight.

        Parameters
        ----------
        to : instance of SimpleNode (or any derived class)
            Node to connect to
        weight : float, optional
            Weight of connection, by default 0
        """
        
        self.connectedTo[to] = weight
        self.degree += 1
        self.total_weight += weight
        
    def removeNeighbour(self, to):
        """Remove any connection (to and from) node @self and node @to.

        Parameters
        ----------
        to : instance of SimpleNode (or any derived class)
            Node to disconnect from. The connection between @self and @to and the one from @to and @self are removed.

        Returns
        -------
        w : float
            Weight of removed connection
        """
        
        if (to in self.connectedTo):
            self.degree -= 1
            w = self.connectedTo[to]
            self.total_weight -= w
            del self.connectedTo[to]
            
            return w
        else:
            return 0
            
    def __str__(self):
        """Return the string representation of the Node, reporting all of its connections."""
        
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
    
    def getConnections(self):
        """Return list of identifiers of nodes connected to @self"""
        
        return [x.id for x in self.connectedTo.keys()]
    
    def getWeights(self):
        """Return the list of weights of connections from @self to other nodes"""
        
        return list(self.connectedTo.values())
    
    def getId(self):
        """Return the @self identifier"""
        
        return self.id
    
    
class Graph:
    """Represents a collections of Nodes connected by weighted edges"""
    
    def __init__(self, adjacency = None, positions = None, node_type = SimpleNode):
        """Create a new empty graph, with nodes instantiated from @node_type class (must inherit from SimpleNode).
        If a (N,N) @adjacency matrix is provided, along with (N,) positions, the graph is automatically filled with N nodes,
        following this specification.

        Parameters
        ----------
        adjacency : ndarray of shape (N,N), optional
            Adjacency matrix for initialization, by default None
        positions : ndarray of shape (N,), optional
            2D positions of nodes (for plotting), by default None
        node_type : class inheriting from SimpleNode, optional
            Class used for instantiating nodes, by default SimpleNode
        """
        
        self.vertList = {} #List of pairs (node identifier, node instance)
        self.numVertices = 0
        self.node_type = node_type #Node type
        
        self.plotting = False #Whether the graph has been plotted at any time. Needed to open the GUI interface only once.
        
        #Initialization
        if (adjacency is not None) and (positions is not None): #Construct graph from matrix representation
            self.fromAdjacency(np.array(adjacency), np.array(positions))
        
    def addVertex(self, key, pos, **kwargs):
        """Add a vertex with identifier @key and 2D pos @pos.

        Parameters
        ----------
        key : hashable
            Node identifier
        pos : ndarray of shape (2,)
            Node's position (for plotting)

        Returns
        -------
        node_type
            The newly created node instance.
        """
        
        self.numVertices += 1
        newVertex = self.node_type(key, pos, **kwargs)
        self.vertList[key] = newVertex
        return newVertex
    
    def getVertex(self, id):
        """Return the node with identifier @id"""
        
        if id in self.vertList:
            return self.vertList[id]
        else:
            return None
    
    def addEdge(self, fromVertex, toVertex, weight=0):
        """Add an edge from @fromVertex to @toVertex (if they are identifiers of existing nodes in the graph) weighted by @weight"""
        
        if (fromVertex in self.vertList) and (toVertex in self.vertList):
            self.vertList[fromVertex].addNeighbour(self.vertList[toVertex], weight)
        else:
            raise ValueError("Cannot create an edge if one or both of the extrema do not exist")
    
    def addSymmetricEdge(self, fromVertex, toVertex, weight=0):
        """Add two edges, representing a 'undirected' link: one from @fromVertex to @toVertex, and the other from @toVertex to @fromVertex (if they are 
        valid identifiers of nodes in the graph). 
        They are both weighted by @weight"""
        
        if (fromVertex in self.vertList) and (toVertex in self.vertList):
            self.vertList[fromVertex].addNeighbour(self.vertList[toVertex], weight)
            self.vertList[toVertex].addNeighbour(self.vertList[fromVertex], weight)
        else:
            raise ValueError("Cannot create an edge if one or both of the extrema do not exist")
    
    def getVertices(self):
        """Returns all identifiers of nodes in the graph"""    
        
        return self.vertList.keys()
    
    def __contains__(self, id):
        return id in self.vertList
    
    def __iter__(self):
        return iter(self.vertList.values())
    
    def __len__(self):
        return self.numVertices
    
    def __str__(self):
        return '\n'.join([str(node) for node in self.vertList.values()])
            
    def fromAdjacency(self, adjacency, positions): 
        """Fill the graph from an @adjacency matrix, positioning the nodes at @positions.

        Parameters
        ----------
        adjacency : ndarray of shape (N,N)
            Adjacency matrix
        positions : ndarray of shape (N,)
            Positions of nodes (for plotting)
        """
        
        #Check adjacency to be a square 2D matrix
        a_shape   = adjacency.shape
        pos_shape = positions.shape
        assert len(a_shape) == 2, "Adjacency matrix must be 2D"
        assert len(pos_shape) == 2, "Positions must be a 2D array"
        
        assert a_shape[0] == a_shape[1], "Adjacency matrix must be a square matrix"
        
        N = a_shape[0]
        assert (pos_shape[0] == N) and (pos_shape[1] == 2), "Positions matrix must be of shape (N,2)"
        
        #Add vertices
        for i, pos in enumerate(positions):
            self.addVertex(i, pos)
        
        #Add edges
        for i, row in enumerate(adjacency):
            for j, weight in enumerate(row):
                if weight != 0:
                    self.addEdge(i, j, weight=weight)
    
    def plot(self, nodeSize=1):
        """Plot the graph with pyqtgraph. @nodeSize is the dimension of nodes in the plot"""
        
        if not self.plotting:
            self.plotting = True
            #Create new plot
            pg.setConfigOptions(antialias = True)
            
            self.w = pg.GraphicsWindow() #Create a window
            self.w.setWindowTitle('Graph plot')
            self.v = self.w.addViewBox()
            self.v.setAspectLocked()

            self.g = pg.GraphItem()
            self.v.addItem(self.g)        
        
        #Update plot
        N = len(self.vertList)  #number of vertices
        positions = np.zeros((N, 2))
        
        edges = []
        weights = []
        colors = []
        borderColors = []
        
        for i, node in enumerate(self.vertList.values()):
            positions[i] = node.position
            
            new_edges = [[i, j] for j in node.getConnections()]
            edges.extend(new_edges)
            weights.extend(node.getWeights())
            colors.append(node.color)
            
            if node.isolated:
                borderColors.append(pg.mkPen(color=(0,0,255), width=5)) #Blue border
            else:
                borderColors.append(pg.mkPen(color=(255,255,255), width=5)) #White border
            
        lines = plt.cm.rainbow(np.array(weights), bytes=True)
    
        self.g.setData(pos=positions, adj=np.array(edges, dtype=int), pen=lines, size=nodeSize, symbol=['o' for i in range(N)], symbolBrush=colors, symbolPen=borderColors, pxMode=False) #pen=lines #symbolSize=500, 
        #If size is too small, some nodes have a weird red square in the background for some reason

    
def test_Graph(): #For testing purposes
    adj_matrix = np.array([[0,1,0,0,0],
                           [1,0,1,0,0],
                           [0,1,0,1,1],
                           [0,0,1,0,1],
                           [0,0,1,1,0]])
    positions = np.array([[0.5,1],
                          [0,0],
                          [1,0],
                          [2,0],
                          [1.5,0.5]])

    g = Graph(adj_matrix, positions)
    
    #Verify structure
    assert g.getVertex(0).getConnections() == [1]
    assert g.getVertex(1).getConnections() == [0, 2]
    assert g.getVertex(2).getConnections() == [1, 3, 4]
    assert g.getVertex(3).getConnections() == [2, 4]
    assert g.getVertex(4).getConnections() == [2, 3]
    
    #Verify degrees
    assert g.getVertex(0).degree == 1
    assert g.getVertex(1).degree == 2
    assert g.getVertex(2).degree == 3
    assert g.getVertex(3).degree == 2
    assert g.getVertex(4).degree == 2 


if __name__ == "__main__":
    #Create a simple graph and plot it
    adj_matrix = np.array([[0,.1,0,0,0],
                           [.1,0,1,0,0],
                           [0,1,0,1,1],
                           [0,0,1,0,1],
                           [0,0,1,1,0]])
    positions = np.array([[0.5,1],
                          [0,0],
                          [1,0],
                          [2,0],
                          [1.5,0.5]])

    g = Graph(adj_matrix, positions)
    
    g.plot(.2)
    
    input("Press ENTER to exit...")