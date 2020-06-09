from Graph import Graph
import Epidemic
import GraphFactory

from pyqtgraph.Qt import QtCore, QtGui
from pyqtreporter import PyQtReporter
from copy import deepcopy
from scipy.special import softmax
import numpy as np
import time
import os
import neat
import configparser
import multiprocessing
import gzip

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

#Save/load objects (used to store network parameters)
def save_object(obj, filename):
    with gzip.open(filename, 'w', compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with gzip.open(filename) as f:
        obj = pickle.load(f)
        return obj

#Retrieve config from parameters file
local_dir = os.path.dirname(__file__)
config = configparser.ConfigParser()
config_rtn = config.read(os.path.join(local_dir, 'parameters.txt'))
config = config["DEFAULT"]

class GeneticNode(Epidemic.SEIR_node):
    """Add actions (isolate / reconnect) to nodes in the graph"""
    
    def __init__(self, identifier, position, state = Epidemic.State.SUSCEPTIBLE):
        super().__init__(identifier, position, state)
        
        #Store "disabled" connections here
        self.old_weight = 0
        self.old_weights = {}
        self.old_connections = {}
        self.old_degree = 0
        
        self.isolated = False #Flag: if True, node is in quarantine (has no connections with other nodes)
    
    def addNeighbour(self, to_id, weight=0):
        #If node is not isolated, the connection is immediately added. Otherwise, it is stored and added only if a "reconnect" action is taken
        if not self.isolated:
            self.connectedTo[to_id] = weight
            self.degree += 1
            self.total_weight += weight
        else:
            self.old_connections[to_id] = weight
            self.old_degree += 1
            self.old_weights[to_id.id] = weight
        
    def isolate(self): 
        """Remove all connections from/to this node"""
        
        #Connections are merely "disabled", and are still stored in a separate variable
        self.isolated = True
        
        if self.degree == 0:
            return 0
        
        #Store connections
        self.old_weight = self.total_weight
        self.old_connections = self.connectedTo #Store old connections
        self.old_degree = self.degree
        
        #Delete all connections to this node
        for nodeTo in self.connectedTo.keys():
            w = nodeTo.removeNeighbour(self)
            self.old_weights[nodeTo.id] = w
        
        self.degree = 0
        self.total_weight = 0
        self.connectedTo = {} #Delete all connections from this node
        
        return self.old_weight
    
    def reconnect(self): 
        """Reconnect the node to the graph"""
        
        #All "disabled" connections are enabled
        self.isolated = False
        
        if (self.old_degree != self.degree):
            self.degree = self.old_degree
            self.connectedTo = self.old_connections
            self.total_weight = self.old_weight

            self.old_degree = 0
            self.old_connections = {}
            self.old_weight = 0
            
            isolated_connections = []
            for nodeTo in self.connectedTo.keys():
                nodeTo.addNeighbour(self, self.old_weights[nodeTo.id])
                
                if nodeTo.isolated:
                    isolated_connections.append(nodeTo)
            
            for iso in isolated_connections:
                self.removeNeighbour(iso)
                
            self.old_weights = {}
    
    #Compute observations (inputs for RNN)
    def getInfectedWeight(self):
        infected_weight = 0
        for node, w in self.connectedTo.items():
            if node.state is Epidemic.State.INFECTED:
                infected_weight += w
        return infected_weight
    
    def getTotalWeight(self):
        return sum(list(self.connectedTo.values()))
        

class Simulation:
    """Interface with NEAT. Stores the environment (Graph with GeneticNodes) and retrieves all the necessary configuration."""
    
    def __init__(self, graph):
        self.g = deepcopy(graph) #Store a local copy of the graph
        self.checkpoint = deepcopy(graph)
        self.runs_per_net = int(config["RunsPerNet"])
        self.max_steps_per_run = int(config["MaxStepsPerRun"])
        self.N = self.g.numVertices
        
        self.timeout = int(config["ExposedAverageDuration"]) + 1 #If number of infected does not increase in a certain number of steps, terminate the run
        
        #---Network options---#
        self.N_inputs = 3
    
        self.obs = np.zeros((self.N, self.N_inputs)) #Array to store observations
        
        self.num_edges = 0
        for node in self.g:
            self.num_edges += len(node.getConnections())
        self.num_edges /= 2 #Undirected graph
        
        removable_fraction = float(config["MaxRemovableConnections"])
        self.removable_edges = int(removable_fraction * self.num_edges)
        
    def step_sim(self, nets):
        #---Retrieve observations (RNN inputs)---#
        
        #Observations are:
        #   0: total infected weight (i.e. sum over weights of connected nodes that are INFECTED)
        #   1: personal state, can be 0 (not INFECTED) or 1 (INFECTED)
        #   2: fraction of infected nodes in the graph
        
        for i, node in enumerate(self.g):
            if node.degree == 0:
                self.obs[i,0] = 0
            else:
                self.obs[i,0] = node.getInfectedWeight() 
            
            self.obs[i,1] = int(node.state is Epidemic.State.INFECTED)

        num_infected = sum(self.obs[:,1])
        self.obs[:,2] = num_infected / self.N #Number of infected in the graph
        
        #---Compute actions---#
        outputs = np.array([nets[i].activate(self.obs[i]) for i in range(self.N)])
        actions = np.argmax(outputs, axis=1) #Highest output is chosen (deterministic, makes training easier for the genetic algorithm)

        #---Execute actions---#
        for action, node in zip(actions, self.g):
            if action == 0: 
                if node.isolated: #Call node methods only if necessary
                    pass
                else:
                    node.isolate()
            elif action == 1:
                if node.isolated:
                    node.reconnect()
                else:
                    pass
        #Note that to keep a node isolated for N timesteps, the "isolate" action must be taken *at each* timestep! (In this case there is no difference as there are only two actions possible, but in principle this should be the behaviour)
        
        #---Step evolution---#        
        num_connections = 0
        for node in self.g:
            node.step()
            num_connections += len(node.getConnections()) 
        num_connections /= 2 #Undirected graph
        
        #---Cost function (=-fitness)---#
        return num_infected**2 + max((self.num_edges - num_connections - self.removable_edges),0) #Return metrics to minimize
    
    def reset(self):
        """Reset the graph to the initial state"""
        
        self.g = deepcopy(self.checkpoint)
        
    def eval_genome(self, genome, net_config):
        """Generate a RNN for each node according to @genome and @neat_config.
        Simulate the systems according to the provided configuration, and return the fitness of that genome."""
        
        runs_metrics = []
            
        for i in range(self.runs_per_net):
            self.reset()
            
            #Create a network for each node
            nets = [neat.nn.RecurrentNetwork.create(genome, net_config) for n in range(self.N)]
            
            metric_to_minimize = 0 #Cost
            
            for t in range(self.max_steps_per_run):
                metric_to_minimize += self.step_sim(nets) #Accumulate cost
                
            runs_metrics.append(metric_to_minimize)
                   
        return -max(runs_metrics) 
        
    def eval_genomes(self, genomes, net_config):
        """Evaluates a list of genomes"""
        
        for genome_id, genome in genomes:                   
            genome.fitness = self.eval_genome(genome, net_config)

if __name__ == "__main__":
    
    #Create graph
    gen = GraphFactory.make_graph("PreferentialAttachment")
    g = gen(30, .5, seed=42, nodetype=GeneticNode) #Starting graph
    
    #Add some infected
    g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    #Advance epidemic for 2 days before letting the algorithm intervene.
    np.random.seed(42)
    for i in range(2):
        for node in g:
            node.step()
    
    #Load NEAT configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat')
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(neat_config)
    
    p.add_reporter(neat.StdOutReporter(True))
    
    # pyqt = PyQtReporter() # Add live plot of fitness. Note that this (for some reason) produces an error when trying to save the final model. So only uncomment these lines during testing!
    # p.add_reporter(pyqt)
    
    #Create interface with graph
    sim = Simulation(g)
    
    #Evaluate for 100 epochs
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), sim.eval_genome)
    winner = p.run(pe.evaluate, 10)
    
    #Save the results #Comment these lines when using the pyqtreporter!
    save_object(winner, "WinnerParams")
    save_object(neat_config, "NeatConfig")
    save_object(g, "OriginalGraph")
    
    #---Show the resulting behaviour---#
    N = g.numVertices
    winner_nets = [neat.nn.RecurrentNetwork.create(winner, neat_config)
                   for i in range(N)]
    
    sim.reset() #Reset sim
    sim.g.plot()
    
    def update(): #Step the simulation
        sim.step_sim(winner_nets)
        sim.g.plot()

    #Calls update() every 500ms to update the graph
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(500)
    
    #Print final results
    print('\nBest genome:\n{!s}'.format(winner))
    
    input("Press Enter to exit...")
    
    # pyqt.proc.close() # Uncomment if pyqtreporter is used
    
    
    
        
        
    