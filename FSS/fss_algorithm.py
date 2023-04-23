import numpy as np
import random
import math
from numpy.linalg import norm
from tqdm.auto import tqdm 

class FSS(object):
    
    def __init__(self):
        self.init
        self.fish_init_
        self.fish_step
        self.delta_f
        self.weight
        self.instinctive_collective_step
        self.coll_step
        
    def fish_init_(self, coord, populationSize):
        x, y, z = coord[0], coord[1], coord[2]
        x_1 = []
        y_1 = []
        z_1 = []
        
        i_mass = []
        j_mass = []

        for a in range(0, populationSize):

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
    
    
    def fish_step(self, coord, i_j, individStep):
        
        a = random.randint(-1,1)
        b = random.randint(-1,1)
        
        x_new = []
        y_new = []
        z_new = []
        i_new = []
        j_new = []
        
        x, y, z = coord[0], coord[1], coord[2]
        i, j = i_j[0], i_j[1]
        
        for step in range(0,len(i)):
            
            if (i[step]+a < len(z) and j[step]+b < len(z) and i[step]+a >= 0 and j[step]+b >= 0):
                
                if z[i[step]+a][j[step]+b] < z[i[step]][j[step]]:
                    
                    x_new.append(x[i[step]+a][j[step]+b])
                    y_new.append(y[i[step]+a][j[step]+b])
                    z_new.append(z[i[step]+a][j[step]+b])
                    i_new.append(i[step]+a)
                    j_new.append(j[step]+b)
                    
                else:
                    x_new.append(x[i[step]][j[step]])
                    y_new.append(y[i[step]][j[step]])
                    z_new.append(z[i[step]][j[step]])
                    i_new.append(i[step])
                    j_new.append(j[step])
                    
            else:
                    
                x_new.append(x[i[step]][j[step]])
                y_new.append(y[i[step]][j[step]])
                z_new.append(z[i[step]][j[step]])
                i_new.append(i[step])
                j_new.append(j[step])
                    
        coord_new = np.array([x_new, y_new, z_new])
        i_j_new = np.array([i_new, j_new])
        
                    
        return coord_new, i_j_new
    
    
    def delta_f(self, z_coord):
        
        delta = []
        
        for i in range(0, len(z_coord[0])):
            z_ = (z_coord[0][i] - z_coord[1][i])
            delta.append(z_)
            
        return delta
    
    def weight(self, z_coord, weightScale):
        
        delta = self.delta_f(z_coord = z_coord)
        weight_mass = []
        
        delta_max = np.max(delta)
        
        if delta_max == 0:
            delta_max = 1
        
        if type(weightScale) == int:
            
            for i in range(0, len(z_coord[0])):
                
                d = (delta[i]/delta_max)+weightScale

                weight_mass.append(d)
                
        else:
            for i in range(0, len(z_coord[0])):
                
                d = (delta[i]/delta_max)+weightScale[i]

                weight_mass.append(d)

        
        return weight_mass, delta
        
            
        
    def instinctive_collective_step(self, coord, i_j, z_coord):
        
        delta = self.delta_f(z_coord)
        d = []
        
        delt = np.sum(delta)
        if delt == 0:
            delt = 0.1
        
        
        for i in range(len(delta)):
            
            d.append(delta[i]**2)
            
        m = np.sum(d)/delt
        m = int(m)
        
        x_new = []
        y_new = []
        z_new = []
        i_new = []
        j_new = []
        
        x, y, z = coord[0], coord[1], coord[2]
        i, j = i_j[0], i_j[1]
        
        for step in range(0,len(i)):
            
            if (i[step]+m < len(z) and j[step]+m < len(z) and i[step]+m >= 0 and j[step]+m >= 0):
                
                if z[i[step]+m][j[step]+m] < z[i[step]][j[step]]:
                    
                    x_new.append(x[i[step]+m][j[step]+m])
                    y_new.append(y[i[step]+m][j[step]+m])
                    z_new.append(z[i[step]+m][j[step]+m])
                    i_new.append(i[step]+m)
                    j_new.append(j[step]+m)
                    
                else:
                    x_new.append(x[i[step]][j[step]])
                    y_new.append(y[i[step]][j[step]])
                    z_new.append(z[i[step]][j[step]])
                    i_new.append(i[step])
                    j_new.append(j[step])
                    
            else:
                    
                x_new.append(x[i[step]][j[step]])
                y_new.append(y[i[step]][j[step]])
                z_new.append(z[i[step]][j[step]])
                i_new.append(i[step])
                j_new.append(j[step])
                    
        coord_new = np.array([x_new, y_new, z_new])
        i_j_new = np.array([i_new, j_new])
        
                    
        return coord_new, i_j_new
        
    def barycenter(self, z_coord, weight):
        
        z_wight = []
        
        for i in range(0, len(z_coord)):
            
            z_wight.append(z_coord)
        
        z_sum = np.sum(z_wight)
        sum_weight = np.sum(weight)
        
        bary = z_sum/sum_weight
        
        return bary
    
    
    def coll_step(self, coord, i_j, z_coord, individStep, bary_center):
        
        coord_new = []
        
        for i in range(len(i_j[1])):
            
            a = random.randint(0,1)
            
            step = (individStep**2)*a*((z_coord[2][i] - bary_center)/(norm(z_coord[2][i] - bary_center)))
            
            if math.isnan(step) == True:
                coord_new.append(0)
            else:
                coord_new.append(round(step))
            
        x_new = []
        y_new = []
        z_new = []
        i_new = []
        j_new = []
        
        x, y, z = coord[0], coord[1], coord[2]
        i, j = i_j[0], i_j[1]
        
        for step in range(0,len(i)):
            
            if (i[step]+coord_new[step] < len(z) and j[step]+coord_new[step] < len(z) and i[step]+coord_new[step] >= 0 and j[step]+coord_new[step] >= 0):
                
                if z[i[step]+coord_new[step]][j[step]+coord_new[step]] < z[i[step]][j[step]]:
                    
                    x_new.append(x[i[step]+coord_new[step]][j[step]+coord_new[step]])
                    y_new.append(y[i[step]+coord_new[step]][j[step]+coord_new[step]])
                    z_new.append(z[i[step]+coord_new[step]][j[step]+coord_new[step]])
                    i_new.append(i[step]+coord_new[step])
                    j_new.append(j[step]+coord_new[step])
                    
                else:
                    x_new.append(x[i[step]][j[step]])
                    y_new.append(y[i[step]][j[step]])
                    z_new.append(z[i[step]][j[step]])
                    i_new.append(i[step])
                    j_new.append(j[step])
                    
            else:
                    
                x_new.append(x[i[step]][j[step]])
                y_new.append(y[i[step]][j[step]])
                z_new.append(z[i[step]][j[step]])
                i_new.append(i[step])
                j_new.append(j[step])
                    
        coord_new = np.array([x_new, y_new, z_new])
        i_j_new = np.array([i_new, j_new])
        
                    
        return coord_new, i_j_new
    
    
    def init(self, x, y, z, populationSize, iterationCount, individStep, weightScale):
        
        coord_mass = []
        coord_mean_mass = []
        coord_min_mass = []
        
        coord_fish_start, i_j_coord = self.fish_init_(coord = [x,y,z], populationSize = populationSize) # inizialisation
        
        coord_fish_one = coord_fish_start
        
        coord_mass.append(coord_fish_start)
        coord_mean_mass.append(np.mean(coord_fish_start[2]))

        for i in tqdm(range(0,iterationCount)):
            
            coord_fish, i_j_coord = self.fish_step(coord = [x, y, z], i_j = i_j_coord, individStep = individStep)

            weightScale, delta_mass = self.weight(z_coord = [coord_fish_one[2], coord_fish[2]], weightScale = weightScale)
            
            coord_fish_one, i_j_coord = self.instinctive_collective_step(coord = [x, y, z], i_j = i_j_coord, z_coord = [coord_fish_one[2], coord_fish[2]])
            
            weightScale, delta_mass = self.weight(z_coord = [coord_fish_one[2], coord_fish[2]], weightScale = weightScale)
            
            bary_c = self.barycenter(z_coord = coord_fish_one, weight = weightScale)
            
            coord_fish = coord_fish_one
            
            coord_fish_one, i_j_coord = self.coll_step(coord = [x, y, z], i_j = i_j_coord, z_coord = coord_fish_one, individStep = individStep, bary_center = bary_c)
            
            coord_mass.append(coord_fish_one)
            coord_mean_mass.append(np.mean(coord_fish_one[2]))
            coord_min_mass.append(np.mean(coord_fish_one[2]))
            
        
        coord_mass = np.array(coord_mass)
        min_coord_z = np.min(coord_mass[len(coord_mass)-1][2])
        
        
        return coord_mass, coord_mean_mass, min_coord_z
        
            


        
            

