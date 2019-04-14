__author__ = 'dora'

import pickle

class MyCells:
    def __init__(self,name = None):
        self.name = name
        self.cells = []
    def createCells(self, animal = None, day = None, px = None,py = None,pt = None, arena_radius = None, hd = None, environment = None, speeds = None, name = None, sx = None, sy = None, st = None, region = None, headdir = None, F = None, binsize = None, spatial_info = None, grid_score = None, spatial_autocorrelation = None, K = None, period = None, sx_all = None, sy_all =None, st_all = None, hd_score = None, theta_timing = None, theta_cycle_parameters =  None, spike_phases = None, st_autocorr = None, phenotype = None, class_label = None, border_score = None, spike_labels = None, spike_distances = None, field_centers = None):
        cell = Cell(animal, day, px, py, pt, arena_radius, hd, environment, speeds, name, sx, sy, st, region, headdir, F, binsize, spatial_info, grid_score, spatial_autocorrelation,K,period, sx_all, sy_all, st_all, hd_score,theta_timing, theta_cycle_parameters, spike_phases, st_autocorr, phenotype, class_label, border_score, spike_labels, spike_distances, field_centers)
        self.cells.append(cell)
    def save_object(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class Cell(MyCells):
    #nrofcells=0
    def __init__(self, animal = None, day = None, px = None,py = None,pt = None, arena_radius = None, hd = None, environment = None, speeds = None, name = None, sx = None, sy = None, st = None, region = None, headdir = None, F = None, binsize = None, spatial_info = None, grid_score = None, spatial_autocorrelation = None, K = None, period = None, sx_all = None, sy_all =None, st_all = None, hd_score = None, theta_timing = None, theta_cycle_parameters =  None, spike_phases = None, st_autocorr = None, phenotype = None, class_label = None, border_score = None, spike_labels = None, spike_distances = None, field_centers = None):
        self.animal = animal
        self.day = day
        self.px = px
        self.py = py
        self.pt = pt
        self.arena_radius = arena_radius
        self.hd = hd
        self.environment = environment
        self.speeds = speeds
        self.name = name
        self.sx = sx
        self.sy = sy
        self.st = st
        self.region = region
        self.headdir = headdir
        self.F = F
        self.binsize = binsize
        self.spatial_info = spatial_info
        self.grid_score = grid_score
        self.spatial_autocorrelation = spatial_autocorrelation
        self.K = K
        self.period = period
        self.sx_all = sx_all
        self.sy_all = sy_all
        self.st_all = st_all
        self.hd_score = hd_score
        self.theta_timing = theta_timing
        self.theta_cycle_parameters = theta_cycle_parameters
        self.spike_phases = spike_phases
        self.st_autocorr = st_autocorr
        self.phenotype = phenotype
        self.class_label = class_label
        self.border_score = border_score
        self.spike_labels = spike_labels
        self.spike_distances = spike_distances
        self.field_centers = field_centers


        #Session.nrofcells +=1

    def save_object(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class MyCellsS:
    def __init__(self,name = None):
        self.name = name
        self.cells = []
    def createCells(self, animal = None, day = None, px = None,py = None,pt = None, arena_radius = None, hd = None, environment = None, speeds = None, name = None, sx = None, sy = None, st = None, region = None, headdir = None, spike_labels = None):
        cell = CellS(animal, day, px, py, pt, arena_radius, hd, environment, speeds, name, sx, sy, st, region, headdir,spike_labels)
        self.cells.append(cell)
    def save_object(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class CellS(MyCellsS):
    #nrofcells=0
    def __init__(self, animal = None, day = None, px = None,py = None,pt = None, arena_radius = None, hd = None, environment = None, speeds = None, name = None, sx = None, sy = None, st = None, region = None, headdir = None, spike_labels = None):
        self.animal = animal
        self.day = day
        self.px = px
        self.py = py
        self.pt = pt
        self.arena_radius = arena_radius
        self.hd = hd
        self.environment = environment
        self.speeds = speeds
        self.name = name
        self.sx = sx
        self.sy = sy
        self.st = st
        self.region = region
        self.headdir = headdir
        self.spike_labels = spike_labels


        #Session.nrofcells +=1

    def save_object(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)