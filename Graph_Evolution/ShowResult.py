import Genetic
import numpy as np
import GraphFactory
import gzip
import Epidemic
import neat
from pyqtgraph.Qt import QtCore, QtGui

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
    # gen = GraphFactory.make_graph("PreferentialAttachment")
    # g = gen(100, .5, seed=42, nodetype=Genetic.GeneticNode) 
    
    # Create Graph
    gen = GraphFactory.make_graph("KarateClub") #graph from real data
    g = gen(nodetype=Genetic.GeneticNode)
    
    #Add some infected
    g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    #Advance epidemic for 3 days without action (to introduce more exposed)
    np.random.seed(42)
    for i in range(3):
        for node in g:
            node.step()
    
    #Load pre-trained network
    neat_config = load_object("NeatConfig")
    winner = load_object("WinnerParams")
    
    #---Simulation---#
    N = g.numVertices
    winner_nets = [neat.nn.RecurrentNetwork.create(winner, neat_config)
                   for i in range(N)]
    
    sim = Genetic.Simulation(g)
    
    sim.g.plot()
    
    input("Press ENTER to start...")
    def update():
        sim.step_sim(winner_nets)
        sim.g.plot()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)
    
    print('\nBest genome:\n{!s}'.format(winner))
    
    input("Press Enter to continue...")
    pyqt.proc.close()
    
    #TODO Add a plot of infected/time