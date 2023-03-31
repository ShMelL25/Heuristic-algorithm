import numpy as np
import pandas as pd
import random

class FSS(object):
    
    def __init__(self):
        
        self.fish
        self.fish_init
        self.weight
        self.step_fish
    
    
    def fish(self, x, y, z, populationSize=100, iterationCount=10, lowerBoundPoint=None, higherBoundPoint=None, individStepStart=None, individStepFinal=None, weightScale=1000.0):
        
        x_fish, y_fish, z_fish, i_j = self.fish_init(x = x, y = y, z = z, populationSize=100)
        
        weight_fish = self.weight(x = x_fish, y = y_fish, z = z_fish, i_j = i_j, weightScale = weightScale)
        
        x_fish_new, y_fish_new, z_fish_new, i_j_new = self.step_fish(x, y, z, i = i_j[0], j = i_j[1])
        
        return i_j, i_j_new
        
        
    # initialization fish
    def fish_init(self, x, y, z, populationSize):
        x_1 = []
        y_1 = []
        z_1 = []
        
        i_mass = []
        j_mass = []

        for a in range(1, populationSize):

            i = random.randint(0, len(x)-1)
            j = random.randint(0, len(x[i])-1)

            x_1.append(x[i][j])
            y_1.append(y[i][j])
            z_1.append(z[i][j])
            
            i_mass.append(i)
            j_mass.append(j)
        
        i_j = np.array([i_mass, j_mass])
        
        return x_1, y_1, z_1, i_j
        
    #fish weight
    def weight(self,x, y, z, i_j, weightScale):
        
        weight_fish=weightScale/2
        
        return weight_fish
    
    #step fish
    def step_fish(self, x, y, z, i, j):
        
        x_new_coord = []
        y_new_coord = []
        z_new_coord = []
        i_new = []
        j_new = []
        
        
        for step in range(0,len(i)):
                
            if ((i[step] != 99) and (j[step] != 99)) and ((i[step] != 0) and (j[step] != 0)):
                a = random.randint(-1,1)
                b = random.randint(-1,1)

                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif ((i[step] == 99) and (j[step] == 99)) and ((i[step] != 0) and (j[step] != 0)):
                    
                a = random.randint(-1,0)
                b = random.randint(-1,0)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                
            elif (i[step] == 99) and (j[step] == 0):
                    
                a = random.randint(-1,0)
                b = random.randint(0,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif (i[step] == 0) and (j[step] == 99):
                    
                a = random.randint(0,1)
                b = random.randint(-1,0)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif (i[step] == 99) and (j[step] != 0):
                    
                a = random.randint(-1,0)
                b = random.randint(-1,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif (j[step] == 99) and (i[step] != 0):
                    
                b = random.randint(-1,0)
                a = random.randint(-1,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                
                    
            elif (i[step] == 0) and (j[step] == 0):
                    
                a = random.randint(0,1)
                b = random.randint(0,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif (i[step] == 0) and (j[step] != 99):
                        
                a = random.randint(0,1)
                b = random.randint(-1,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
            elif (j[step] == 0) and (i[step] != 99):
                    
                a = random.randint(-1,1)
                b = random.randint(0,1)
                    
                x_new_coord.append(x[i[step]+a][j[step]+b])
                y_new_coord.append(y[i[step]+a][j[step]+b])
                z_new_coord.append(z[i[step]+a][j[step]+b])
                i_new.append(i[step]+a)
                j_new.append(j[step]+b)
                    
        
        i_j_new = np.array([i_new, j_new])
        return x_new_coord, y_new_coord, z_new_coord, i_j_new
        

def f(x1,x2):
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    
    return c

FSS_alg = FSS()

x1=np.linspace(-512,512,100)
x2=np.linspace(-512,512,100)
X1,X2=np.meshgrid(x1,x2)

c = f(X1,X2)
i, j = FSS_alg.fish(x=X1, y=X2, z=c)

print(i)
print(j)