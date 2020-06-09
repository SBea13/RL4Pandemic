from Graph import Graph, SimpleNode
from enum import IntEnum
import configparser
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import os

#Possible states of Infection
class State(IntEnum):
    SUSCEPTIBLE  = 1
    EXPOSED      = 2
    INFECTED     = 3
    RECOVERED    = 4

#Colors of each state (for plotting)
State.colors = {
    State.SUSCEPTIBLE : "#4bf521", #Green
    State.EXPOSED     : "#f8e936", #Yellow
    State.INFECTED    : "#ff0000", #Red
    State.RECOVERED   : "#8d8d8d"  #Gray
}

#Retrieve config from parameters file
local_dir = os.path.dirname(__file__)
config = configparser.ConfigParser()
config_rtn = config.read(os.path.join(local_dir, 'parameters.txt'))
config = config["DEFAULT"]

if config_rtn == []:
    raise ValueError("Missing configuration")

class SEIR_node(SimpleNode):
    """Add SEIR Evolution to nodes in the graph"""
    
    def __init__(self, identifier, position, state = State.SUSCEPTIBLE): #By default, nodes are SUSCEPTIBLE
        super().__init__(identifier, position)
    
        self.setState(state)
        self.counter = 0 #Add time counter (for evolution)
    
    def setState(self, new_state):
        self.state = new_state
        self.color = State.colors[self.state]
    
    def step(self):
        """SEIR Evolution of @self node"""
        
        rnd = np.random.uniform() 
        
        if self.state == State.SUSCEPTIBLE:
            #Compute weight of connections with infected/exposed nodes
            infected_weight = 0
            asymptomatic_weight = 0
            asymptomatic_multiplier = float(config["AsymptomaticMultiplier"])
            activation_speed = float(config["ActivationSpeed"])
            
            for node in self.connectedTo:
                if node.state == State.INFECTED:
                    infected_weight += self.connectedTo[node]
                if node.state == State.EXPOSED:
                    asymptomatic_weight += self.connectedTo[node] * asymptomatic_multiplier
            
            if config["InfectionActivationFunction"] == "tanh":
                p_infection = np.tanh(activation_speed * (infected_weight + asymptomatic_weight))
            else:
                raise NotImplementedError()
            
            if rnd <= p_infection:
                self.state = State.EXPOSED
                
        elif self.state == State.EXPOSED: #Deterministic evolution EXPOSED -> INFECTED
            self.counter += 1
            if self.counter == int(config["ExposedAverageDuration"]):
                self.state = State.INFECTED
                self.counter = 0
        
        elif self.state == State.INFECTED: #Deterministic evoluton INFECTED -> RECOVERED
            self.counter += 1
            
            if self.counter == int(config["InfectedAverageDuration"]):
                self.state = State.RECOVERED
                self.counter = 0
        else:
            pass
        
        #Update color
        self.color = State.colors[self.state]


if __name__ == "__main__":
    #Create a sample graph and compute a SEIR evolution over it
    
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

    g = Graph(adj_matrix, positions, node_type=SEIR_node)
    
    g.getVertex(3).setState(State.INFECTED)
    
    g.plot(.2)
    
    def update():
        print("Stepping!")
        for node in g:
            node.step()
        g.plot(.2)
        
        
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(300)
    
    input("Press ENTER to continue...")

