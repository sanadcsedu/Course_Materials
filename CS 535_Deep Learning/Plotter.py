# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:03:25 2019

@author: Sanad Saha
"""
import numpy as np
import matplotlib.pyplot as plt

def plot(x_axis, y_axis, xstep = 1.0, file = None):
        
        
        plt.xticks(np.arange(min(x_axis), max(x_axis), xstep))

        plt.plot(x_axis, y_axis, '-r')
        
        plt.ylabel('Total Time')
        plt.xlabel('Number of CPU Cores')
        plt.title('Total Running Time Vs. Number of CPUs')
        plt.legend(loc='best')
        if file is not None:
            plt.savefig('ML/%s' % file, bbox_inches='tight')
        plt.gcf().clear()
        
if __name__ == '__main__':
    
    x = [1, 2, 4, 8]
    #reduce time
    #y = [2.111182451248169, 1.2068696022033691, 0.9061517715454102,  0.9085993766784668]
    
    #total time
    y = [6.122781276702881, 3.2143497467041016, 1.9117233753204346, 1.914504051208496]
    #y = [6.12278, 3.21435, 1.911723, 1.91450]
    
    plot(x, y, 1.0, 'CPU_Vs_Time2.jpg')