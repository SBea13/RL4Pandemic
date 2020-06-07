from Graph import Graph
from Graph import SimpleNode
from enum import IntEnum
import configparser
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

class State(IntEnum):
    SUSCEPTIBLE  = 1
    EXPOSED      = 2
    INFECTED     = 3
    RECOVERED    = 4
    
State.colors = {
    State.SUSCEPTIBLE : "#4bf521", #Green
    State.EXPOSED     : "#f8e936", #Yellow
    State.INFECTED    : "#ff0000", #Red
    State.RECOVERED   : "#8d8d8d"  #Gray
}
    
config = configparser.ConfigParser()
config_rtn = config.read('parameters.txt')
config = config["DEFAULT"]

if config_rtn == []:
    raise ValueError("Missing configuration")


class SEIR_node(SimpleNode):
    def __init__(self, identifier, position, state = State.SUSCEPTIBLE):
        super().__init__(identifier, position)
    
        self.setState(state)
        self.counter = 0 #Add time counter (for evolution)
    
    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
    
    def setState(self, new_state):
        self.state = new_state
        self.color = State.colors[self.state]
    
    def step(self):
        #Compute evolution
        rnd = np.random.uniform()
        
        if self.state == State.SUSCEPTIBLE:
            #Compute weight of connections with infected nodes
            infected_weight = 0
            asymptomatic_weight = 0
            asymptomatic_multiplier = float(config["AsymptomaticMultiplier"])
            activation_speed = float(config["ActivationSpeed"])
            for node in self.connectedTo:
                if node.state == State.INFECTED:
                    infected_weight += self.connectedTo[node]
                if node.state == State.EXPOSED:
                    asymptomatic_weight += self.connectedTo[node] * asymptomatic_multiplier
                    
            #print("I'm ", self.id, " weight: ", infected_weight)
            
            if config["InfectionActivationFunction"] == "tanh":
                p_infection = np.tanh(activation_speed * (infected_weight +                                 asymptomatic_weight))
                #Use a tanh activation function (Make it configurable)
            else:
                raise NotImplementedError()
            
            if rnd <= p_infection:
                self.state = State.EXPOSED
                #print(self.id, "got exposed!")
                
        elif self.state == State.EXPOSED:
            self.counter += 1
            if self.counter == int(config["ExposedAverageDuration"]):
                self.state = State.INFECTED
                self.counter = 0
        
        elif self.state == State.INFECTED:
            self.counter += 1
            
            if self.counter == int(config["InfectedAverageDuration"]):
                self.state = State.RECOVERED
                self.counter = 0
        
        else:
            pass
        
        #Update color
        self.color = State.colors[self.state]


if __name__ == "__main__":
    # adj_matrix = np.array([[0,.1,0,0,0],
    #                        [.1,0,1,0,0],
    #                        [0,1,0,1,1],
    #                        [0,0,1,0,1],
    #                        [0,0,1,1,0]])
    # positions = np.array([[0.5,1],
    #                       [0,0],
    #                       [1,0],
    #                       [2,0],
    #                       [1.5,0.5]])

    # g = Graph(adj_matrix, positions, node_type=SEIR_node)
    
    # g.getVertex(3).setState(State.INFECTED)
    
    #Generate random graph
    N = 50
    positions = np.random.uniform(-10, 10, size=(N, 2))
    adjacency = np.random.uniform(0, 1, size=(N,N))
    
    adjacency[adjacency < .9] = 0 #remove weak connections
    
    adj = (adjacency.T + adjacency) / 1.4 #symmetrize
    
    g = Graph(adj, positions, node_type=SEIR_node)
    
    g.getVertex(0).setState(State.INFECTED)
    
    g.plot()
    
    def update():
        #print("Stepping!")
        for node in g:
            node.step()
        g.plot()
        
        
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(300)
    
    input("Press ENTER to continue...")

