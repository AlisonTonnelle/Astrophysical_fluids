# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:29:08 2023

@author: Alison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Ngrid =  100
# x,dx = np.linspace()


Nsteps = 5000
# dt = 1e-6            #the time step as well as the spatial step ensure that Courant condition is satisfied
dx = 1

x = np.arange(0.1, Ngrid*1., dx) / Ngrid


#Setting up the velocity and the two boundary conditions for velocity

# nu = 10**3*10**(-4) #m²/s
nu = 1
v = 9*nu/(2*x)

v[0] = -abs(v[1])
v[-1] = -abs(v[-2])

# v = -0.1

dt = dx/(max(v))/10
# dt = 1
 
alpha = v*dt/(2*dx)

print(alpha)
#Initial condition

sigma = signal.gaussian(len(x),2.) 
# sigma = np.copy(x)  #Gaussian function at the middle of the plot

f = np.copy(sigma)    #advection part for the beginning
T = np.copy(sigma)    #diffusion part for the beginning


#Plot settings

plt.ion()
fig, ax = plt.subplots(1,1)
ax.plot(x,sigma,'k-')          #plot of initial condition
evp, = ax.plot(x,sigma,'r')   #plot which will evolve over time

#Axis limits

ax.set_xlim([0,1])
ax.set_ylim([0,1])

fig.canvas.draw()



#Setup diffusion part
n = Ngrid
D = 3*nu
beta = D*dt/(dx**2)
A = np.eye(n) *(1.0 + 2.0 * beta) + np.eye(n,k=1) * (-beta) + np.eye(n, k=-1) * (-beta)


#Evolution

for ct in range(Nsteps): #using Lax-Friedrichs for the advection part
    f[1:-1] = 0.5*(f[2:] + f[:-2]) - (alpha[2:]*f[2:] - alpha[:-2]*f[:-2])
    # f[1:-1] = 0.5*(f[2:] + f[:-2]) - alpha*(f[2:] - f[:-2])
    T = np.linalg.solve(A,T)
    sigma[1:-1] = f[1:-1] + T[1:-1]
    sigma[0] = sigma[1]
    sigma[-1] = sigma[-2]
    
    
    evp.set_ydata(sigma)
    
    fig.canvas.draw()
    plt.pause(0.001)
    
