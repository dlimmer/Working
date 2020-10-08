import numba, sys
from pylab import *
from numpy import random
import matplotlib as mpl
mpl.style.use('classic')

# Initialize N particles on a
# a square lattice of length Lx
# with Maxwell Boltzmann velocities
@numba.jit(nopython=True)
def initialize(x,v,L,Nx):
    dx=L/Nx
    dx=L/Nx
    sigma=T
    n=0
    for i in range(Nx):
        for j in range(Nx):
            x[n,0]=i*dx
            x[n,1]=j*dx
            v[n,0]=np.random.normal(0,sigma)
            v[n,1]=np.random.normal(0,sigma)
            n=n+1
    v[:,0]=v[:,0]-sum(v[:,0])/(n-1)
    v[:,1]=v[:,1]-sum(v[:,1])/(n-1)

    return x,v

# Compute the pair forces acting on each particle
@numba.jit(nopython=True)
def force(x,f,Lx):

    for i in range(N):
        f[i,0]=0.
        f[i,1]=0.
        for j in range(N):
            if i==j: continue
            dx=x[i,0]-x[j,0]
            dx=dx-Lx*np.round(dx/Lx)
            dy=x[i,1]-x[j,1]
            dy=dy-Lx*np.round(dy/Lx)
            r=np.sqrt(dx*dx+dy*dy)
            if r<1.12246204831: #2**(1./6.)
                fac=4*(12./r**14.-6./r**8.)
                f[i,0]=f[i,0]+fac*dx
                f[i,1]=f[i,1]+fac*dy

    return f


# Integrate Newton's equations
@numba.jit(nopython=True)
def Verlet(x,v,f,f1,Lx):

    f=force(x,f,Lx)
    x+=v*dt+f*dt**2./2.
    f1=force(x,f1,Lx)
    v+=(f+f1)*dt/2.
    x=x%Lx
    return x,v

# Parameters #
##
# Size of lattice, Nx x Nx
Nx = 8
# Total number of particles
N = Nx*Nx
# Temperature
T=2.5
# Size of Box
Lx=10.
# Timestep
dt=0.002
# Area
A=Lx**2
# Number of steps
steps = 5000
# Averaging frequency
Nsamp = 10

print("MD Run with N=", N, "for time=", steps*dt, "and area density,", N/A)

# Container for positions in 2d
x=np.zeros([N,2])
# Container for velocities in 2d
v=np.zeros([N,2])
# Container for forces in 2d
f=np.zeros([N,2])
f1=np.zeros([N,2])
# Initialize lattice
x,v=initialize(x,v,Lx,Nx)

imflag=1

if imflag:
    fig, ax = plt.subplots(1, 1,figsize=(8,8))
    ax.set_aspect('equal')
    ax.set_xlim(0, Lx-1/2.)
    ax.set_ylim(0, Lx-1/2.)
    plt.setp(ax, xticks=[], yticks=[])
    points = ax.plot(x[:,0],x[:,1],'ob',ms=23*20./Lx)[0]
    draw()

# Preform steps integrations
for step in range(steps):

  # Perform N random displacements
  x,v=Verlet(x,v,f,f1,Lx)
  
  # Every Nsamp steps, compute expectations
  if(step%Nsamp==0 and step>0):

   if imflag:
     if step==Nsamp: keyboardClick=plt.waitforbuttonpress()
            
     points.set_data(x[:,0],x[:,1])
     fig.canvas.draw()
     plt.pause(0.02)

if raw_input("<Hit Enter To Close>"):
    plt.close(fig)

if imflag:
 plt.savefig('HardSphere.png')
 plt.close()


