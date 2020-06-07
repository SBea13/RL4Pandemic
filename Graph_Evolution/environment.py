# -*- coding: utf-8 -*-
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class State(Enum):
    SUSCEPTIBLE  = 1
    ASYMPTOMATIC = 2
    INFECTED     = 3
    RECOVERED    = 4

# State.colors = {
#     State.SUSCEPTIBLE  :  [76, 235, 52, 255], #Green
#     State.ASYMPTOMATIC :  [235, 229, 52, 255], #Yellow
#     State.INFECTED : [235, 52, 52, 255], #Red
#     State.RECOVERED : [128, 128, 128, 255] #Gray
# }

    
    
    
        

class Graph:
    def __init__(self):
        self.N = 10
        self.positions = np.random.uniform(-5,5, size=(self.N, 2)) #Positions of nodes
        
        #Connect each node with random weights
        self.adj_matrix = np.random.uniform(0, 1, size=(self.N, self.N))
        self.adj_matrix = (self.adj_matrix + self.adj_matrix.T) / 2 #make it symmetric
        self.adj_matrix[self.adj_matrix < .5] = 0 #Remove weak connections
        np.fill_diagonal(self.adj_matrix, 0) #Remove self-connections
        
        self.node_attributes = {}
        
        #Node attributes
        self.add_node_attribute("state", np.random.randint(1,4+1, size=self.N))
        
        self.node_colors = np.array(["#000000", "#4bf521", "#f8e936", "#ff0000", "#8d8d8d"])
                                      #Void    #Susceptible #Asymptomatic #Infected #Removed
                                      
    def add_node_attribute(self, name, init=None):
        if init is not None:
            #Check if name already exists
            if name in self.node_attributes.keys():
                raise ValueError("Node attribute already present")
            if len(init) != self.N:
                raise ValueError("Not enough values in initialization")
            
            self.node_attributes[name] = np.array(init)
        else:
            self.node_attributes[name] = np.zeros(self.N)

    
    def get_node_attribute(self, name):
        if name not in self.node_attributes.keys():
            self.add_node_attribute(name)
        
        return self.node_attributes[name]
        
    def compute_edges(self):
        upper_triangular = np.triu_indices(self.N)
        all_edges = np.array(upper_triangular).T
        
        matrix_upper = self.adj_matrix[upper_triangular]

        return (all_edges[matrix_upper > 0], matrix_upper[matrix_upper > 0])
        
    def plot(self):
        pg.setConfigOptions(antialias = True)
        
        self.w = pg.GraphicsWindow() #Create a window
        self.w.setWindowTitle('Graph plot')
        self.v = self.w.addViewBox()
        self.v.setAspectLocked()

        self.g = pg.GraphItem()
        self.v.addItem(self.g)        
        
        edges, weights = self.compute_edges()
        weights = np.random.uniform(0, 1, size=len(edges))
        lines = plt.cm.rainbow((weights + 1)/2, bytes=True)
        
        points_color = self.node_colors[self.get_node_attribute("state")]
        
        #points_color = np.array(['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'r'])
        
        #print(adj.dtype)
        #print(edges)
        print(lines.shape)
        print(points_color.shape)
        self.g.setData(pos=self.positions, adj=edges, pen=lines, size=.4, symbol=['o' for i in self.positions], symbolBrush=points_color, pxMode=False) #pen=lines #symbolSize=500, 
        

def test_compute_edges():
    g = Graph()
    g.adj_matrix = np.array([[0,1,0,0,0],
                             [1,0,1,0,0],
                             [0,1,0,1,1],
                             [0,0,1,0,1],
                             [0,0,1,1,0]])
    g.N = 5
    g.positions = np.array([[0.5,1],
                            [0,0],
                            [1,0],
                            [2,0],
                            [1.5,0.5]])
                           
    edges, _ = g.compute_edges()
    
    assert np.all(edges == np.array([[0,1], [1,2], [2,3], [2,4], [3,4]]))
    
    g.plot()
    
    input("Write something...")

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    #import sys
    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #    QtGui.QApplication.instance().exec_()
    
    g = Graph()
    g.plot()
    
    input("Inserisci qualcosa...")
