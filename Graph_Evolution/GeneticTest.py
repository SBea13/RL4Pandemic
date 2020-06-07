import Epidemic
import Genetic
import GraphFactory
import Graph


if __name__ == "__main__":
    # gen = GraphFactory.make_graph("PreferentialAttachment")
    # g = gen(11, .5, seed=42, nodetype=Genetic.GeneticNode) #Starting graph
    
    # #Add some infected
    # g.getVertex(0).setState(Epidemic.State.INFECTED)
    
    # edges = 0
    # for node in g:
    #     edges += len(node.getConnections())
    # edges /= 2 #Undirected graph
    # print("There are", edges, "edges")
    
    # print(g.getVertex(0))
    # g.plot(1)
    # input("Press ENTER to continue...")
    
    # g.getVertex(0).isolate()
    
    # print(g.getVertex(0))
    # g.plot(1)
    # input("Press ENTER to continue...")
    
    # g.getVertex(0).reconnect()
    
    # print(g.getVertex(0))
    # g.plot(1)
    
    g = Graph.Graph(node_type=Genetic.GeneticNode)
    
    v0 = g.addVertex(0, [0,0])
    v1 = g.addVertex(1, [1,1])
    v2 = g.addVertex(2, [1,0])
    v3 = g.addVertex(3, [-1,0])
    
    g.addEdge(0, 1, weight=.5)
    g.addEdge(1, 0, weight = .5)
    
    g.addEdge(1, 2, weight=.5)
    g.addEdge(2, 1, weight=.5)
    
    g.addEdge(0, 2, weight=.5)
    g.addEdge(2, 0, weight=.5)
    
    g.addEdge(3, 0, weight=.5)
    g.addEdge(0, 3, weight=.5)
    
    print(g)
    g.plot(.3)
    
    assert set(v0.getConnections()) == set([1,2,3])
    assert set(v2.getConnections()) == set([0,1])
    
    input("Press ENTER to continue...")
    
    v0.isolate()
    print("0 isolated!")
    
    assert v0.getConnections() == []
    assert set(v2.getConnections()) == set([1])
    
    g.plot(.3)
    input("Press ENTER to continue...")
    
    v2.isolate()
    print("2 isolated!")
    assert v0.getConnections() == []
    assert v2.getConnections() == []
    
    g.plot(.3)
    input("Press ENTER to continue...")
    
    v0.reconnect()
    print("0 reconnected!")
    assert set(v0.getConnections()) == set([1,3])
    assert v2.getConnections() == []
    
    g.plot(.3)
    input("Press ENTER to continue...")
    
    
    v2.reconnect()
    print("2 reconnected!")
    assert set(v0.getConnections()) == set([1,2,3])
    assert set(v2.getConnections()) == set([0,1])
    
    g.plot(.3)
    input("Press ENTER to continue...")
    
    
    
    