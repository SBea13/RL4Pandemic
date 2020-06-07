import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

class SimpleNode:    
    """Represents a node in a graph"""
    
    def __init__(self, identifier, position):
        self.id = identifier
        self.position = position
        self.connectedTo = {}
        
        
        self.degree = 0
        self.total_weight = 0
        
        self.color = "#FFFFFF"  #Color for plotting (white default)
        
        self.isolated = False
    
    def addNeighbour(self, to_id, weight=0):
        """[summary]

        Parameters
        ----------
        to_id : class Node
            [description]
        weight : int, optional
            [description], by default 0
        """
        self.connectedTo[to_id] = weight
        self.degree += 1
        self.total_weight += weight
        
    def removeNeighbour(self, to_id):
        if (to_id in self.connectedTo):
            self.degree -= 1
            w = self.connectedTo[to_id]
            self.total_weight -= w
            del self.connectedTo[to_id]
            #print("Removed", to_id.id)
            
            return w
        else:
            return 0
            
    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
    
    def getConnections(self):
        return [x.id for x in self.connectedTo.keys()]
    
    def getWeights(self):
        return list(self.connectedTo.values())
    
    def getId(self):
        return self.id
    
    def getWeight(self, nbr):
        return self.connectedTo[nbr]
    
    
class Graph:
    def __init__(self, adjacency = None, positions = None, node_type = SimpleNode):
        self.vertList = {} #List of vertices
        self.numVertices = 0
        
        self.node_type = node_type #Node type
        
        self.plotting = False
        
        if (adjacency is not None) and (positions is not None): #Construct graph from matrix representation
            self.fromAdjacency(np.array(adjacency), np.array(positions))
        
    def addVertex(self, key, pos): #Add kwargs
        self.numVertices += 1
        newVertex = self.node_type(key, pos)
        self.vertList[key] = newVertex
        return newVertex
    
    def getVertex(self, id):
        if id in self.vertList:
            return self.vertList[id]
        else:
            return None
    
    def __contains__(self, id):
        return id in self.vertList
    
    def addEdge(self, fromVertex, toVertex, weight=0):
        if (fromVertex in self.vertList) and (toVertex in self.vertList):
            self.vertList[fromVertex].addNeighbour(self.vertList[toVertex], weight)
        else:
            raise ValueError("Cannot create an edge if one or both of the extrema do not exist")
    
    def addSymmetricEdge(self, fromVertex, toVertex, weight=0):
        if (fromVertex in self.vertList) and (toVertex in self.vertList):
            self.vertList[fromVertex].addNeighbour(self.vertList[toVertex], weight)
            self.vertList[toVertex].addNeighbour(self.vertList[fromVertex], weight)
        else:
            raise ValueError("Cannot create an edge if one or both of the extrema do not exist")
    
    def getVertices(self):
        return self.vertList.keys()
    
    def __iter__(self):
        return iter(self.vertList.values())
    
    def __len__(self):
        return self.numVertices
    
    def __str__(self):
        return '\n'.join([str(node) for node in self.vertList.values()])
            
    def fromAdjacency(self, adjacency, positions): 
        """Construct the graph from an adjacency matrix"""
        
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
            

        #print(colors)

        lines = plt.cm.rainbow(np.array(weights), bytes=True)
    
        self.g.setData(pos=positions, adj=np.array(edges, dtype=int), pen=lines, size=nodeSize, symbol=['o' for i in range(N)], symbolBrush=colors, symbolPen=borderColors, pxMode=False) #pen=lines #symbolSize=500, 
        #If size is too small, some nodes have a weird red square in the background, idk why
            
        #Strange square plotted 
#Create a testing function        
def test_Graph():
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
    
    #TODO: Add more tests 


if __name__ == "__main__":
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
    
    g.plot()
    
    input("Press ENTER to exit...")