#!/usr/bin/env python
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt
plt.matplotlib.use('TkAgg')
plt.rcParams['axes.linewidth'] = 4

def randomwalk(dims=(100, 100), n=1000, sigma=1.5, alpha=.5, seed=34234):
    """ A simple random walk with memory """

    r, c = dims
    gen = np.random.RandomState(seed)
    #pos = gen.rand(2, n) * ((r,), (c,))
    pos=np.zeros([2,n])
    pos[0,:]=100/2
    pos[1,:]=99 #100/2
    #old_delta = gen.randn(2, n) * sigma

    while True:
        delta = (1. - alpha) * gen.randn(2, n) * sigma #+ alpha * old_delta
        pos += delta
        #pos[0,:]=pos[0,:]%100
        pos[0,pos[0,:]>99]=pos[0,pos[0,:]>99]-2*delta[0,pos[0,:]>99]
        pos[0,pos[0,:]<1.]=pos[0,pos[0,:]<1.]-2.*delta[0,pos[0,:]<1.]
        pos[1,pos[1,:]>99]=pos[1,pos[1,:]>99]-2*delta[1,pos[1,:]>99]
        pos[1,pos[1,:]<1]=pos[1,pos[1,:]<1]-2.*delta[1,pos[1,:]<1]
        #pos[1,:]=pos[1,:]%100
        yield pos


def run(niter=10000, doblit=True):
    """
    Display the simulation using matplotlib, optionally using blit for speed
    """
    fig, ax = plt.subplots(1, 1,figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 102)
    ax.spines['top'].set_visible(False)
    ax.hold(True)
    rw = randomwalk()
    x, y = rw.next()
    plt.setp(ax, xticks=[], yticks=[])
    a=np.arange(0,100,.1)
    plt.plot(a,99.+np.sin(.10*a+np.sin(.30*a))/10.,'-k',lw=.5)
    plt.show(False)
    plt.draw()

    tag=1
    
    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

    if len(x[:])<20: points = ax.plot(x, y, '.',alpha=1,ms=20)[0]
    elif len(x[:])<10000: points = ax.plot(x, y, '.',alpha=.5,ms=14)[0]
    else: points = ax.plot(x, y, '.',alpha=.1,ms=2)[0]
    
    if tag: points2 = ax.plot(x[:5], y[:5], '.r',alpha=0.5,ms=10)[0]

    tic = time.time()

    for ii in xrange(niter):

        # update the xy data
        x, y = rw.next()
        points.set_data(x, y)
        if tag: points2.set_data(x[:5], y[:5])
        
        if doblit:
            # restore background
            fig.canvas.restore_region(background)
            # redraw just the points
            ax.draw_artist(points)
            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)

        if ii%10==0:
            # redraw everything
            fig.canvas.draw()
            plt.pause(0.0001)

        if ii==0:
            if raw_input("<Hit Enter To Start>"):
                plt.pause(1)


    plt.pause(0.02)
    if raw_input("<Hit Enter To Close>"):
        plt.close(fig)

if __name__ == '__main__':
    run(doblit=False)

