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
    if tagflag:
        v[:,:]=0.
        v[10,0]=6.2*2
        v[-1,0]=-6.2*2
        v[10,1]=7.2343*2
        v[-1,1]=-7.2343*2

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
            if r<1.112:
                fac=4*(12./r**14.-6./r**8.)
                f[i,0]=f[i,0]+fac*dx
                f[i,1]=f[i,1]+fac*dy

    return f

# If desired compute the potential energy
@numba.jit(nopython=True)
def potential(x,Lx):
    pe=0.
    for i in range(N):
        for j in range(N):
            if i==j: continue
            dx=x[i,0]-x[j,0]
            dx=dx-Lx*np.round(dx/Lx)
            dy=x[i,1]-x[j,1]
            dy=dy-Lx*np.round(dy/Lx)
            r=np.sqrt(dx*dx+dy*dy)
            if r<1.112:
                pe=pe+4.*(1./r**12.-1./r**6.)+1.
    return pe/2.


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

# Make image of system
imflag = True

# Make velocity distribution
vhistflag= True

# Make speed distribution
vdhistflag= False

# Make occupation distribution
dhistflag= False
Rstar=2.
Nm=int(Rstar**2.*pi/Lx/Lx*N)
NNs=array([])

# Initialize two particles with nonzero v
tagflag= True
# Integrate backwards
backflag= True

print("MC Run with N=", N, "Number of MC steps=", steps, "and area density,", N/A)

# Container for positions in 2d
x=np.zeros([N,2])
# Container for velocities in 2d
v=np.zeros([N,2])
vtot=array([])
vtot2=array([])
# Container for forces in 2d
f=np.zeros([N,2])
f1=np.zeros([N,2])
# Initialize lattice
x,v=initialize(x,v,Lx,Nx)


if imflag:
    if vhistflag:
        fig1, ax1 = plt.subplots(1, 1,figsize=(8,8))
        points3=ax1.hist(v[:,0],bins=arange(-8,8,1),density=1)
        draw()
    if vdhistflag:
        fig1, ax1 = plt.subplots(1, 1,figsize=(8,8))
        points3=ax1.hist(sqrt(v[:,0]**2.+v[:,1]**2.),bins=arange(0,8,1),density=1)
        draw()

    if dhistflag:
        fig1, ax1 = plt.subplots(1, 1,figsize=(8,8))
        RR=sqrt((x[:,0]-Lx/2.)**2.+(x[:,1]-Lx/2.)**2.)
        NNs=append(NNs,len(RR[RR<Rstar]))
        points3=ax1.hist(NNs,bins=arange(0,5*Nm,1),normed=1)
        ax1.plot(Lx/2.,Lx/2.,'or',20.,alpha=0.2)
        draw()
    fig, ax = plt.subplots(1, 1,figsize=(8,8))
    ax.set_aspect('equal')
    ax.set_xlim(0, Lx-1/2.)
    ax.set_ylim(0, Lx-1/2.)
    plt.setp(ax, xticks=[], yticks=[])
    points = ax.plot(x[:,0],x[:,1],'ob',ms=23*20./Lx)[0]
    if dhistflag:
        s = ((ax.get_window_extent().width  / (Lx-1/2.) * 72./fig.dpi) ** 2)
        ax.scatter(Lx/2.,Lx/2.,s=s*10*Rstar,alpha=0.2)
        ax.scatter(Lx/2.,Lx/2.,s=s*10*Rstar*2,alpha=0.2)
        ax.scatter(Lx/2.,Lx/2.,s=s*10*Rstar*3.5,alpha=0.2)

    if tagflag: points2 = ax.plot([x[10,0],x[-1,0]],[x[10,1],x[-1,1]],'or',ms=23*20./Lx)[0]
    draw()


# Preform steps integrations
for step in range(steps):

  # Perform N random displacements
  x,v=Verlet(x,v,f,f1,Lx)
  
  if backflag==1 and step==steps/2:
    x,v=Verlet(x,-v,f,f1,Lx)
  
  # Every Nsamp steps, compute expectations
  if(step%Nsamp==0 and step>0):
   
   pe=potential(x,Lx)
   vtot=append(vtot,v[:,0])
   vtot2=append(vtot2,sqrt(v[:,0]**2.+v[:,1]**2.))

   if imflag:
     if step==Nsamp: keyboardClick=plt.waitforbuttonpress()

     points.set_data(x[:,0],x[:,1])
     if tagflag: points2.set_data([x[10,0],x[-1,0]],[x[10,1],x[-1,1]])
     fig.canvas.draw()
     plt.pause(0.02)

     if(step%(20*Nsamp)==0 and vhistflag):
        print(step)
        ax1.cla()
        ax1.hist(vtot,bins=arange(-8,8,1),density=1)
        l=arange(-10,10,.01)
        m=var(vtot)
        pv=exp(-l**2./2./m)/sqrt(2*pi*m)
        ax1.plot(l,pv,'--r',lw=4)
        
        fig1.canvas.draw()
        plt.pause(0.02)
     
     if(step%(20*Nsamp)==0 and vdhistflag):
         ax1.cla()
         ax1.hist(vtot2,bins=arange(0,8,1),density=1)
         l=arange(0,10,.01)
         m=var(vtot)
         pv=l**2.*exp(-l**2./2./m)/sqrt(pi/2.)/m**1.5
         pv=l*exp(-l**2./2./m)/m
         ax1.plot(l,pv,'--r',lw=4)
         
         fig1.canvas.draw()
         plt.pause(0.02)

     if(step%(20*Nsamp)==0 and dhistflag):
        ax1.cla()
        RR=sqrt((x[:,0]-Lx/2.)**2.+(x[:,1]-Lx/2.)**2.)
        NNs=append(NNs,len(RR[RR<Rstar]))
        ax1.hist(NNs,bins=arange(0,5*Nm,1),density=1)
        fig1.canvas.draw()
        plt.pause(0.02)


if raw_input("<Hit Enter To Close>"):
    plt.close(fig)

if imflag:
 plt.savefig('HardSphere.png')
 plt.close()


