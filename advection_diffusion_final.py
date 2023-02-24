# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:29:08 2023

@author: Alison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Ngrid =  100         
dx = 0.01

#Definition of the x axis
x = np.linspace(0,1,Ngrid)

#Setting up the velocity and the two boundary conditions for velocity
nu = 10**3*10**(-4) #m²/s seen in an article

v = -9*nu/(2*x)                        #expression of velocity given with the analytical calculation
v[0] = -abs(v[1])                      #boundary condition
v[-1] = abs(v[-2])                     #same

dt = abs((dx/max(v)))/90              #the time step which ensures that Courant 
tf = 5.                                #condition is satisfied, divided by 10 to have a margin + visualization
Nsteps = int(tf/dt)        
                   
#Initial condition
sigma = signal.gaussian(len(x),2.)      #Gaussian function at the middle of the plot

f = np.copy(sigma)                      #advection part for the beginning
T = np.copy(sigma)                      #diffusion part for the beginning


#Setup diffusion part
n = Ngrid
D = 3.*nu                               #expression found with the formal expression
beta = D*dt/(dx**2)
A = np.eye(n) *(1.0 + 2.0 * beta) + np.eye(n,k=1) * (-beta) + np.eye(n, k=-1) * (-beta)


#Plot settings
plt.ion()
fig, ax = plt.subplots(1,1)
ax.plot(x,sigma,'k-')                   #plot of initial condition
evp, = ax.plot(x,sigma,'r')             #plot which will evolve over time

#Axis limits
ax.set_xlim([0,1])
ax.set_ylim([0,1])

fig.canvas.draw()


#Evolution

for ct in range(Nsteps): #using Lax-Friedrichs for the advection part

    #Advection part
    f[1:Ngrid-1] = 0.5*(f[2:] + f[:Ngrid-2]) - v[1:Ngrid-1]*dt/(2*dx)*(f[2:] - f[:Ngrid-2])
    
    #Diffusion part
    T = np.linalg.solve(A,T)
    
    #Sum of both contributions
    sigma[1:-1] = f[1:-1] +  T[1:-1]
    
    #Boundary conditions for the density
    sigma[0] = sigma[1]                 
    sigma[-1] = sigma[-2]               
    
    evp.set_ydata(sigma)
    
    fig.canvas.draw()
    plt.pause(0.001)
    
