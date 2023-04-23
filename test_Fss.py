import numpy as np
from heuristic_algorithm import FSS_algorithm, plot_3d

def f(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    
    
    
    return c

x1=np.linspace(-512,512,100)
x2=np.linspace(-512,512,100)
X1,X2=np.meshgrid(x1,x2)
c = f(X1,X2)
mass_coord, mean_, min_ = FSS_algorithm(x=X1, y=X2, z=c, populationSize=300, iterationCount=10, individStep=1)
print(min_)

plot_3d([X1, X2, c], mass_coord)