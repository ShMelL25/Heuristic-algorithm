import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Plot_viz(object):
    
    def __init__(self):
        self.plotter

    def plotter(self, coord_plot, point_coord): 
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(coord_plot[0],coord_plot[1],coord_plot[2], color='orange',alpha=0.2)
        ax.scatter(point_coord[len(point_coord)-1][0], point_coord[len(point_coord)-1][1], point_coord[len(point_coord)-1][2], color='green')
        ax.plot_wireframe(coord_plot[0],coord_plot[1],coord_plot[2],ccount=2,rcount=2, color='red',alpha=0.3)   
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()