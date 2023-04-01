import numpy as np
import pandas as pd
import random

class FSS(object):
    
    def __init__(self):
        
        self.fish
        self.fish_init
        self.weight
        self.step
        self.step_fish
    
    
    def fish(self, x, y, z, populationSize=100, iterationCount=10, lowerBoundPoint=None, higherBoundPoint=None, individStepStart=None, individStepFinal=None, weightScale=1000.0):
        
        coord_fish_iter, i_j = self.fish_init(x = x, y = y, z = z, populationSize=100)
        
        fish_new_coord, i_j_new, weight_new = self.step(x = x, y = y, z = z, i = i_j[0], j = i_j[1], weightScale = weightScale)
        
        return coord_fish_iter, fish_new_coord, weight_new
        
        
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
        
        fish_coord = np.array([x_1, y_1, z_1])
        i_j_coord = np.array([i_mass, j_mass])
        
        return fish_coord, i_j_coord
        
    
    def weight(self, z_mass, weightScale):
        
        weight_fish = weightScale/2
        wieght_iter_fish = []
        
        for i in range(0, len(z_mass[0])):
            
            wieght_iter_fish.append((z_mass[0][i] - z_mass[1][i]/ max(z_mass[0])) + weight_fish)
            
        wieght_iter_fish = np.array(wieght_iter_fish)    
        
        return wieght_iter_fish
    
    def step(self, x, y, z, i, j, weightScale):
        
        x_new_coord = []
        y_new_coord = []
        z_new_coord = []
        i_new = []
        j_new = []
        weigth_fish_iter = []
        
        
        for step in range(0,len(i)):
                
            if ((i[step] != 99) and (j[step] != 99)) and ((i[step] != 0) and (j[step] != 0)):
                a = random.randint(-1,1)
                b = random.randint(-1,1)
                
                x_step, y_step, z_step, i_step, j_step = self.step_fish(x = x[i[step]][j[step]], y = y[i[step]][j[step]], z = z[i[step]][j[step]], 
                                                   i = i[step], j = j[step], x_new = x[i[step]+a][j[step]+b], 
                                                   y_new = y[i[step]+a][j[step]+b], z_new = z[i[step]+a][j[step]+b], i_new = i[step]+a, j_new = j[step]+b)
                
                x_new_coord.append(x_step)
                y_new_coord.append(y_step)
                z_new_coord.append(z_step)
                i_new.append(i_step)
                j_new.append(j_step)
                    
            
            else:
                
                x_new_coord.append(x[i[step]][j[step]])
                y_new_coord.append(y[i[step]][j[step]])
                z_new_coord.append(z[i[step]][j[step]])
                i_new.append(i[step])
                j_new.append(j[step])
                
        
        weigth_fish_iter = self.weight(z_mass = [z_new_coord, z], weightScale = weightScale)            
        fish_coord = np.array([x_new_coord, y_new_coord, z_new_coord])
        i_j_new = np.array([i_new, j_new])
        
        return fish_coord, i_j_new, weigth_fish_iter
    
    
    def step_fish(self, x, y, z, i, j,  x_new, y_new, z_new, i_new, j_new):
        
        if z > z_new:
            return x_new, y_new, z_new, i_new, j_new
        
        else:
            return x, y, z, i, j
            
        
    
FSS_alg = FSS()