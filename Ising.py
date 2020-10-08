from pylab import *
import numba

# Define functions

# Compute energy
@numba.jit(nopython=True)
def get_energy(spins):
 temp=0
 for x in range(Nx):
  for y in range(Nx):
   up=spins[x,(y+1)%Nx]
   down=spins[x,(y-1)%Nx]
   left=spins[(x-1)%Nx,y]
   right=spins[(x+1)%Nx,y]
   temp+=-J*spins[x,y]*(up+down+left+right)/2.-h*spins[x,y]

 return temp 

# Monte carlo loop
@numba.jit(nopython=True)
def monte_carlo(spins,energy,mag):

 for i in range(N):
  x=randint(0,Nx)
  y=randint(0,Nx)
  spins,energy,mag=flip(x,y,spins,energy,mag)

 return spins, energy, mag


# Attempt spin flip with Metropolis acceptance
@numba.jit(nopython=True)
def flip(x,y,spins,energy,mag):
   spino=spins[x,y]
   spinn=-1*spins[x,y]

   up=spins[x,(y+1)%Nx]
   down=spins[x,(y-1)%Nx]
   left=spins[(x-1)%Nx,y]
   right=spins[(x+1)%Nx,y]

   # Compute change in energy
   deltaE =-2.*J*(up+down+left+right+h)
  
   # Metropolis Acceptance Criteria
   if np.random.rand()<exp(-beta*detalE):
    # If move accepted, change spin
    spins[x,y]=spinn
    # Change energy and magnetization
    energy+=deltaE
    mag+=spinn-spino

   return spins, energy, mag


# Number of spins on one side 
Nx=20
# Total number of spins
N=Nx*Nx

# Container for spins

# Start all up
#spins=zeros([Nx,Nx])-1
#spins[0,0]=1

# Random initial condition
spins=randint(0,2,[Nx,Nx])*2-1

# kB Temperature
kT=2.22
# beta
beta=1./kT
# Exchange energy
J=1.
# External field
h=0.0

# Number of MC steps
nsteps=50000
# Print out nstat
nstat=10

# Containers for averages
t_energy=zeros(nsteps)
t_mag=zeros(nsteps)

# Instantaneous values of energy and mag
energy=get_energy(spins)
mag=sum(spins)

# Flag for visualization
imflag=0

if imflag:
    fig, ax = plt.subplots(1, 1,figsize=(8,8))
    ax.set_aspect('equal')
    ax.set_xlim(-.5, Nx-1/2.)
    ax.set_ylim(-.5, Nx-1/2.)
    plt.setp(ax, xticks=[], yticks=[])
    img = ax.imshow(spins,cmap='winter',interpolation='None')
    plt.draw()

print    
print "MC simulation"
print "kT = %.2f h = %.2f J = %.2f" %(kT,h,J)
print "Nx = %d by Nx = %d" %(Nx,Nx)
print "Nsteps = %d and Nstat = %d" %(nsteps,nstat)
print
print "Step \t E \t <E> \t M \t <M>"
print "%d \t %.4f \t %.4f \t %.4f \t %.4f" %(0, energy/N, energy/N, mag/N, mag/N)


for step in range(nsteps):

  spins,energy,mag=monte_carlo(spins,energy,mag)
  t_energy[step]=energy
  t_mag[step]=mag
  if step%nstat==0 and step>0: 
    if imflag:
     img.set_data(spins)
     fig.canvas.draw()
     plt.pause(0.05)
    print "%d \t %.4f \t %.4f \t %.4f \t %.4f" %(step, energy/N, mean(t_energy[:step])/N, t_mag[step]/N, mean(t_mag[:step])/N)

# Show image
imshow(spins,cmap='winter',interpolation='None')
savefig('image.png')
show()
clf()

# Plot magnetization
plot(arange(nsteps),t_mag/N)
xlabel('MC Step')
ylabel(r'$M/N$')
show()

