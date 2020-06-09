from neat.math_util import mean, stdev, median2
from neat.reporting import BaseReporter
import numpy as np
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp
import copy

#Reporting class
#Taken from: https://neat-python.readthedocs.io/en/latest/_modules/statistics.html#StatisticsReporter
#And modified to show a live plot during genetic training, both of fitness and species
class PyQtReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """
    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes = []
        self.generation_statistics = []
        self.pens = 0
        
        #Create pyqt window
        pg.mkQApp() #Must create a QApplication before starting QtProcess
        self.proc = mp.QtProcess()
        rpg = self.proc._import('pyqtgraph')
        
        self.app = rpg.Qt.QtGui.QApplication([])
        self.win = rpg.GraphicsWindow(title="Basic plotting examples")
        self.win.resize(2000, 1000)
        self.win.setWindowTitle('Test')
        rpg.setConfigOptions(antialias=True) #enable antialiasing
        
        p1 = self.win.addPlot(title="Statistics")
        p1.addLegend() 
        
        
        self.curve_up = p1.plot(pen='y')
        self.curve_down = p1.plot(pen='y', name = '1 Std')
        fill = rpg.FillBetweenItem(self.curve_up, self.curve_down, (100, 100, 255, 150))
        p1.addItem(fill)
        
        self.curve = p1.plot(pen='g', symbol='x', name='Mean')
        self.curve_max = p1.plot(pen='r', symbol='o', symbolPen='r', symbolBrush=0.5, name='Max')
        
        self.p2 = self.win.addPlot(title="Species")
        
        #self.species_curves = self.p2.plot(pen='b', symbol='x', name='Test')
        self.species_curves = [] #Store pointers to curves in species graph
        
        #Allocates memory
        self.data = self.proc.transfer([]) 
        self.data_max = self.proc.transfer([])
        self.data_up = self.proc.transfer([])
        self.data_down = self.proc.transfer([])
        
        self.species_data = []
        
        rpg.Qt.QtGui.QApplication.instance().exec_()

    def post_evaluate(self, config, population, species, best_genome):
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        for sid, s in species.species.items():
            species_stats[sid] = dict((k, v.fitness) for k, v in s.members.items())
        
        self.generation_statistics.append(species_stats)
        
        #plot data with PyQt
        mean = self.get_fitness_mean()[-1]
        self.data.extend([mean], _callSync='off') #callSync 'off' means that no result is returned (pure data transfer)
        self.curve.setData(y=self.data, _callSync='off')
        
        maximum = self.get_fitness_stat(np.max)[-1]
        self.data_max.extend([maximum], _callSync='off')
        self.curve_max.setData(y=self.data_max, _callSync='off')
        
        sigma = self.get_fitness_stdev()[-1]
        self.data_up.extend([mean + sigma], _callSync='off')
        self.data_down.extend([mean - sigma], _callSync='off')
        
        self.curve_up.setData(y=self.data_up, _callSync='off')
        self.curve_down.setData(y=self.data_down, _callSync='off')
        
        #self.species_curves.setData(y = self.get_fitness_stat(np.max), _callSync='off')

        #Plot species
        species_sizes = self.get_species_sizes()
        curves = np.array(species_sizes).T
        
        stacked = np.cumsum(curves, axis=0)

        for i, _ in enumerate(stacked):
            if (i < len(self.species_curves)): #if curve already exists, just add the new data points
                print("Try to extend curve", i)
                self.species_data[i].extend([stacked[i][-1]], _callSync='off')
                self.species_curves[i].setData(y = self.species_data[i], _callSync='off')
            else: #otherwise, create a new curve 
                #allocate new data
                print('Allocating', stacked[i])
                new_data = self.proc.transfer([])
                new_data.extend(stacked[i])
                self.species_data.append(new_data)
                
                new_curve = self.p2.plot(y = new_data, fillLevel=0, fillBrush=(i,10))
                new_curve.setZValue(10-i)
                self.species_curves.append(new_curve)
                print('Created 1 more curve')
        
        
    def get_fitness_stat(self, f):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat

    def get_fitness_mean(self):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev)

    def get_fitness_median(self):
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(median2)

    def best_unique_genomes(self, n):
        """Returns the most n fit genomes, with no duplication."""
        best_unique = {}
        for g in self.most_fit_genomes:
            best_unique[g.key] = g
        best_unique_list = list(best_unique.values())

        def key(genome):
            return genome.fitness

        return sorted(best_unique_list, key=key, reverse=True)[:n]

    def best_genomes(self, n):
        """Returns the n most fit genomes ever seen."""
        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        """Returns the most fit genome ever seen."""
        return self.best_genomes(1)[0]

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def get_species_fitness(self, null_value=''):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_fitness = []
        for gen_data in self.generation_statistics:
            member_fitness = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
            fitness = []
            for mf in member_fitness:
                if mf:
                    fitness.append(mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness