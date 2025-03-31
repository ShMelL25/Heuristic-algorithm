from FSS.fss_algorithm import FishSchoolSearch
from visualization.visualization_plot import Plot_viz

def FSS_algorithm(x, y, z, populationSize=200, iterationCount=100, individStep = 1, weightScale = 500):
    return FishSchoolSearch().init(x, y, z, populationSize, iterationCount, individStep, weightScale)

def plot_3d(coord_plot, point_coord):
    
    return Plot_viz().plotter(coord_plot, point_coord)
