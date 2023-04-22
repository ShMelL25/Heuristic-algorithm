from FSS.fss_algorithm import FSS
import numpy as np
from visualization.visualization_plot import Plot_viz

def f(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    
    
    
    return c

x1=np.linspace(-512,512,100)
x2=np.linspace(-512,512,100)
X1,X2=np.meshgrid(x1,x2)
c = f(X1,X2)
mass_coord, mean_, min_ = FSS().init(x=X1, y=X2, z=c, populationSize=300, iterationCount=100, individStep=70)
print(min_)

Plot_viz().plotter([X1, X2, c], mass_coord)