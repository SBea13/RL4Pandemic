import Genetic
import numpy as np
import GraphFactory
import gzip
import Epidemic
import neat
from pyqtgraph.Qt import QtCore, QtGui
import matplotlib.pyplot as plt

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

def save_object(obj, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj

#Use this file to visualize / generate plots of pre-trained networks
if __name__ == "__main__":
    gen = GraphFactory.make_graph("PreferentialAttachment")
    g = gen(30, .5, seed=42, nodetype=Genetic.GeneticNode) 
    
    # Create Graph
    # gen = GraphFactory.make_graph("KarateClub") #graph from real data
    # g = gen(nodetype=Genetic.GeneticNode)
    
    #Add some infected
    g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    #Advance epidemic for 3 days without action (to introduce more exposed)
    np.random.seed(42)
    
    susceptible = np.zeros(100)
    infected = np.zeros(100)
    exposed  = np.zeros(100)
    
    for i in range(2):
        for node in g:
            if node.state is Epidemic.State.INFECTED:
                infected[i] += 1
            elif node.state is Epidemic.State.EXPOSED:
                exposed[i] += 1
            elif node.state is Epidemic.State.SUSCEPTIBLE:
                susceptible[i] += 1
        
            node.step()
    
    #Load pre-trained network
    neat_config = load_object("NeatConfig")
    winner = load_object("WinnerParams")
    
    #---Simulation---#
    N = g.numVertices
    print("Number of vertices: ", N)
    
    winner_nets = [neat.nn.RecurrentNetwork.create(winner, neat_config)
                   for i in range(N)]
    
    sim = Genetic.Simulation(g)
    #sim.g.plot()
    
    #input("Press ENTER to start...")
    
    class Updater:
        def __init__(self):
            self.i = 2
        def update(self):
            for node in sim.g:
                if node.state is Epidemic.State.INFECTED:
                    infected[self.i] += 1
                elif node.state is Epidemic.State.EXPOSED:
                    exposed[self.i] += 1
                elif node.state is Epidemic.State.SUSCEPTIBLE:
                    susceptible[self.i] += 1
                
            sim.step_sim(winner_nets)
            #sim.g.plot()
            self.i += 1

    
    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(1000)
    
    upd = Updater()
    for j in range(97):
        upd.update()
    
    
    print(infected)
    print('\nBest genome:\n{!s}'.format(winner))
    
    input("Press Enter to continue...")
    
    #pyqt.proc.close()
    
    # Run in 