import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import os
import re

def animate(folder):
    files = os.listdir(f"Results/{folder}")
    key = lambda string: int(re.search(r"\d+",string)[0])
    bad = []
    for file in files:
        if "airfoil" not in file:
            bad.append(file)
    for file in bad:
        files.remove(file)
    files.sort(key=key)
    fig = plt.figure()
    ax = plt.axes(xlim=(0,1),ylim=(-.1,.1))
    airfoil, = ax.plot([],[])
    af_list = []
    for file in files:
        outdata = np.loadtxt(f"Results/{folder}/{file}")
        af_list.append(outdata)
    def init(): 
        original = "naca0012.dat"
        outdata = np.loadtxt(original)
        airfoil.set_data(outdata[:,0], outdata[:,1]) 
        return airfoil, 
    def event(frame):
        data = af_list[frame]
        airfoil.set_data(data[:,0],data[:,1])
        return airfoil
    anim = animation.FuncAnimation(fig,func=event,frames = np.arange(0,len(af_list)),interval = 1000)
    writergif = animation.PillowWriter(fps=4)
    anim.save(f"Results/{folder}/slsqp.gif",writergif)
