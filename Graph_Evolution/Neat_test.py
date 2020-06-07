import Graph
import Genetic
import Epidemic
import neat
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import os
import multiprocessing
import pyqtreporter


if __name__ == "__main__":
    g = Graph.Graph(node_type=Genetic.GeneticNode)
    
    #---Vertices---#
    v0 = g.addVertex(0, [0,0])
    v1 = g.addVertex(1, [0,1])
    v2 = g.addVertex(2, [1,1])
    
    v3 = g.addVertex(3, [3,0])
    v4 = g.addVertex(4, [4,0])
    v5 = g.addVertex(5, [4,1])
    v6 = g.addVertex(6, [3,1])
    
    v7 = g.addVertex(7, [2,1])
    
    #---Edges---#
    g.addSymmetricEdge(0, 1, .5)
    g.addSymmetricEdge(1, 2, .5)
    g.addSymmetricEdge(0, 2, .5)
    
    g.addSymmetricEdge(2, 7, .5)
    g.addSymmetricEdge(7, 6, .5)
    
    g.addSymmetricEdge(3, 4, .5)
    g.addSymmetricEdge(4, 5, .5)
    g.addSymmetricEdge(5, 6, .5)
    g.addSymmetricEdge(3, 6, .5)
    
    v2.setState(Epidemic.State.INFECTED)
    # v0.isolate()
    
    # g.plot(.3) #Do not plot before network evolution

    #Edit initial condition
    #v1.setState(Epidemic.State.EXPOSED)
    #v2.setState(Epidemic.State.EXPOSED)
    
    # g.plot(.3)
    # input("Test")
    
    #---Neat Config---#    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat')
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(neat_config)
    p.add_reporter(neat.StdOutReporter(True))
    
    pyqt = pyqtreporter.PyQtReporter()
    p.add_reporter(pyqt)
    
    sim = Genetic.Simulation(g)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), sim.eval_genome)
    winner = p.run(pe.evaluate, 10)
    
    #---Simulation---#
    N = g.numVertices
    winner_nets = [neat.nn.RecurrentNetwork.create(winner, neat_config)
                   for i in range(N)]
    
    sim.reset()
    sim.g.plot(.3)
    def update():
        sim.step_sim(winner_nets)
        sim.g.plot(.3)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)
    
    print('\nBest genome:\n{!s}'.format(winner))
    
    input("Press Enter to continue...")
    # pyqt.proc.close()
    
    # def update():
    #     for node in g:
    #         node.step()
    #         g.plot(.3)

    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(300)
    
    # input("Press ENTER to continue...")
    
    