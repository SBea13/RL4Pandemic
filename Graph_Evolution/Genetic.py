from Graph import Graph
import Epidemic
import numpy as np
import GraphFactory
from pyqtgraph.Qt import QtCore, QtGui
import neat
from pyqtreporter import PyQtReporter
import time
import os
from copy import deepcopy
from scipy.special import softmax
import configparser
import multiprocessing
import gzip

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

config = configparser.ConfigParser()
config_rtn = config.read('parameters.txt')
config = config["DEFAULT"]

class GeneticNode(Epidemic.SEIR_node):
    def __init__(self, identifier, position, state = Epidemic.State.SUSCEPTIBLE):
        super().__init__(identifier, position, state)
        self.resources = int(config["InitialResources"])
        
        self.old_weight = 0
        self.old_weights = {}
        self.old_connections = {}
        self.old_degree = 0
        
        self.isolated = False 
    
    def addNeighbour(self, to_id, weight=0):
        if not self.isolated:
            #print("I'm adding a connection to", self.id)
            self.connectedTo[to_id] = weight
            self.degree += 1
            self.total_weight += weight
        else:
            #print("I'm", self.id, "and I'm isolated!")
            self.old_connections[to_id] = weight
            self.old_degree += 1
            self.old_weights[to_id.id] = weight
            
    
        
    def isolate(self): 
        """Remove all connections from/to this node"""
        self.isolated = True
        
        if self.degree == 0:
            return 0
        
        self.old_weight = self.total_weight
        self.old_connections = self.connectedTo #Store old connections
        self.old_degree = self.degree
        
        #Delete all connections to this node
        for nodeTo in self.connectedTo.keys():
            w = nodeTo.removeNeighbour(self)
            #print(w)
            self.old_weights[nodeTo.id] = w
        
        self.degree = 0
        self.total_weight = 0
        self.connectedTo = {} #Delete all connections from this node
        
        return self.old_weight
    
    def reconnect(self): 
        self.isolated = False
        
        if (self.old_degree != self.degree):
            self.degree = self.old_degree
            self.connectedTo = self.old_connections
            #print(self.connectedTo)
            self.total_weight = self.old_weight

            self.old_degree = 0
            self.old_connections = {}
            self.old_weight = 0
            
            #print(self.connectedTo)
            
            isolated_connections = []
            for nodeTo in self.connectedTo.keys():
                nodeTo.addNeighbour(self, self.old_weights[nodeTo.id])
                
                if nodeTo.isolated:
                    isolated_connections.append(nodeTo)
            
            for iso in isolated_connections:
                self.removeNeighbour(iso)
                
            self.old_weights = {}
        
    def genetic_step(self):
        self.resources = self.resources - int(config["ResourceCost"]) \
                         + self.total_weight
        self.step()
        
        return self.resources
    
    
    def remove_random_connections(self, p=.1):
        """Remove edges with probability p""" #TODO To be fixed
        
        if (self.degree == 0):
            return 0
        
        removedEdges = np.random.uniform(0, 1, size=self.degree) < p
        n_removed_connections = sum(removedEdges)
        
        keys_to_remove = np.array(list(self.connectedTo.keys()))[removedEdges]
    
        original_weight = self.total_weight
        
        for key in keys_to_remove:
            weight = self.connectedTo[key]
            self.total_weight -= weight
            
            key.degree -= 1
            key.total_weight -= weight #TODO Bugged!
            del key.connectedTo[self]
            del self.connectedTo[key]
        
        self.degree -= n_removed_connections
    
        # print("Removed {} connections, total weight: {} -> {}".format(
        #     n_removed_connections, original_weight, self.total_weight
        # ))
        
        return original_weight - self.total_weight #Return cost of operation
    
    def getInfectedWeight(self):
        infected_weight = 0
        for node, w in self.connectedTo.items():
            if node.state is Epidemic.State.INFECTED:
                infected_weight += w
        return infected_weight
    
    def getTotalWeight(self):
        return sum(list(self.connectedTo.values()))
        

class Simulation:
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
        #---Retrieve observations---#
        
        #Observations are:
        #   0: fraction of infected weight #Keep fraction or give absolute number?
        #   1: personal state (infected or not)
        #   (2: fraction of remaining connections)
        
        for i, node in enumerate(self.g):
            if node.degree == 0:
                self.obs[i,0] = 0
            else:
                self.obs[i,0] = node.getInfectedWeight() #/ node.getTotalWeight()
            
            self.obs[i,1] = int(node.state is Epidemic.State.INFECTED)

        num_infected = sum(self.obs[:,1])
        self.obs[:,2] = num_infected / self.N #Number of infected in the graph
        
        #---Compute actions---#
        outputs = np.array([nets[i].activate(self.obs[i]) for i in range(self.N)])
        action_proba = softmax(outputs, axis=1) #Probability of taking each action
        
        n_outputs = action_proba.shape[-1]
        
        actions = np.argmax(outputs, axis=1)
        #actions = [np.random.choice(np.arange(n_outputs, dtype=int), p=action_proba[i])
        #           for i in range(self.N)]

        #---Execute actions---#
        for action, node in zip(actions, self.g):
            if action == 0:
                if node.isolated:
                    pass
                else:
                    node.isolate()
            elif action == 1:
                if node.isolated:
                    node.reconnect()
                else:
                    pass

        #---Step evolution---#        
        num_connections = 0
        for node in self.g:
            node.step()
            num_connections += len(node.getConnections()) 
        num_connections /= 2 #Undirected graph
        
        return num_infected**2 + max((self.num_edges - num_connections - self.removable_edges),0) #Return metrics to minimize
    
    def reset(self):
        self.g = deepcopy(self.checkpoint)
        
    def eval_genome(self, genome, net_config):
        runs_metrics = []
            
        for i in range(self.runs_per_net):
            self.reset()
            
            #Create a network for each node
            nets = [neat.nn.RecurrentNetwork.create(genome, net_config) for n in range(self.N)]
            
            metric_to_minimize = 0
            
            for t in range(self.max_steps_per_run):
                metric_to_minimize += self.step_sim(nets)
                
            runs_metrics.append(metric_to_minimize)
                   
        return -max(runs_metrics) 
        
    def eval_genomes(self, genomes, net_config):
        for genome_id, genome in genomes:                   
            genome.fitness = self.eval_genome(genome, net_config)

if __name__ == "__main__":
    gen = GraphFactory.make_graph("PreferentialAttachment")
    g = gen(30, .5, seed=42, nodetype=GeneticNode) #Starting graph
    
    #Add some infected
    g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    #TODO Steps before
    #Step for 3 days
    np.random.seed(42)
    for i in range(2):
        for node in g:
            node.step()
    
    # g.plot()
    
    # input("Test")
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_neat')
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    p = neat.Population(neat_config)
    
    p.add_reporter(neat.StdOutReporter(True))
    
    # pyqt = PyQtReporter()
    # p.add_reporter(pyqt)
    
    sim = Simulation(g)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), sim.eval_genome)
    winner = p.run(pe.evaluate, 100)
    
    #winner = p.run(sim.eval_genomes, 100)
    
    #Save the results
    save_object(winner, "WinnerParams")
    save_object(neat_config, "NeatConfig")
    save_object(g, "OriginalGraph")
    
    #---Simulation---#
    N = g.numVertices
    winner_nets = [neat.nn.RecurrentNetwork.create(winner, neat_config)
                   for i in range(N)]
    
    sim.reset()
    
    sim.g.plot()
    def update():
        sim.step_sim(winner_nets)
        sim.g.plot()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)
    
    print('\nBest genome:\n{!s}'.format(winner))
    
    input("Press Enter to continue...")
    pyqt.proc.close()
    
    
    
        
        
    