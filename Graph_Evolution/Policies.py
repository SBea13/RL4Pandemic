from Graph import Graph
import Epidemic
import numpy as np
import GraphFactory
from pyqtgraph.Qt import QtCore, QtGui

#Isolate infected (Remove all connections to infected nodes)
# _policies = []

# def policy_type(cls):
#     _policies.append(cls)

# def make_policy(type):
#     for gt in _policies:
#         if gt._name == type:
#             return gt
#     raise NotImplementedError('Graphs of type "{}" are not implemented'.format(type))

# @policy_type
def isolate_infected(graph):
    for node in graph:
        if (node.state is Epidemic.State.INFECTED) or (node.state is Epidemic.State.EXPOSED):
            node.isolate()

def isolate_paths(graph):
    for node in graph:
        infected_weight = sum([w for key, weight
                                in node.connectedTo.items()
                                if key.state is Epidemic.State.INFECTED])
        if infected_weight > 0:
            node.isolate()
        

        
if __name__ == "__main__":
    gen = GraphFactory.make_graph("PreferentialAttachment")
    g = gen(50, 3/50, seed=42, nodetype=Epidemic.SEIR_node)
    
    print(g)
    g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    g.plot()
    
    class Updater:
        def __init__(self):
            self.n = 0
        
        def update(self):
            #print("Stepping!")
            for node in g:
                node.step()
            
            if self.n > 3: #Slight lag before starting the policy
                isolate_infected(g) 
            
            g.plot()
            
            print("n = {}".format(self.n))
            self.n += 1
        
    upd = Updater()
    
    timer = QtCore.QTimer()
    timer.timeout.connect(upd.update)
    timer.start(300)
    
    
    input("Press ENTER to exit...")
    
    
