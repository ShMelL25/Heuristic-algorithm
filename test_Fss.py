import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from FSS import FSS
from ipywidgets import interactive

FSS_ = FSS()

def f(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    
    
    
    return c

x1=np.linspace(-512,512,100)
x2=np.linspace(-512,512,100)
X1,X2=np.meshgrid(x1,x2)
c = f(X1,X2)
mass_coord, min_coord = FSS_.fish(x=X1, y=X2, z=c, populationSize=1000, iterationCount=1000, individStep=1)
print(min_coord)

def plotter(E,A): 
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(X1,X2,c, color='orange',alpha=0.2)
    ax.scatter(mass_coord[0][0], mass_coord[0][1], mass_coord[0][2], color='black', alpha=0.2)
    ax.scatter(mass_coord[len(mass_coord)-1][0], mass_coord[len(mass_coord)-1][1], mass_coord[len(mass_coord)-1][2], color='green')
    ax.plot_wireframe(X1,X2,c,ccount=2,rcount=2, color='red',alpha=0.3)   
    ax.view_init(elev=E,azim=A)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')
    plt.show()

iplot=interactive(plotter,E=(-90,90,5),A=(-90,90,5))
iplot