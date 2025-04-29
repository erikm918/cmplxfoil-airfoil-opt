import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import os
import re

def animate(folder,fps):
    files = os.listdir(f"Results/{folder}")
    bad = []
    for file in files:
        if "airfoil" not in file:
            bad.append(file)
    for file in bad:
        files.remove(file)
    if "PENALTY" in folder:
        key = lambda string: (int(re.findall(r"\d+",string)[0]),int(re.findall(r"\d+",string)[1]))
    else:
        key = lambda string: int(re.findall(r"\d+",string)[0])
    fig = plt.figure()
    original = "naca0012.dat"
    original_data = np.loadtxt(original)
    ax = plt.axes(xlim=(0,1),ylim=(-.1,.1))
    airfoil, = ax.plot([],[])
    origfoil, = ax.plot([],[],color = "black",alpha=.4)
    af_list = []
    iterdata = []
    files.sort(key=key)
    for file in files:
        outdata = np.loadtxt(f"Results/{folder}/{file}")
        iterdata.append(re.findall(r"\d+",file))
        af_list.append(outdata)
    def init(): 
        origfoil.set_data(original_data[:,0],original_data[:,1])
        airfoil.set_data(outdata[:,0], outdata[:,1]) 
        fig.suptitle(folder)
        return airfoil, 
    def event(frame):
        data = af_list[frame]
        fig.suptitle(folder)
        origfoil.set_data(original_data[:,0],original_data[:,1])
        airfoil.set_data(data[:,0],data[:,1])
        iteration = iterdata[frame]
        if "PENALTY" in folder:
            if ax.get_legend():
                ax.get_legend().remove()
            fig.legend([f"Subproblem {iteration[0]}, iter:{iteration[1]}"])
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            fig.legend([f"Iteration {iteration[0]}"])
        return airfoil
    anim = animation.FuncAnimation(fig,func=event,frames = np.arange(0,len(af_list)),interval = 1000)
    writergif = animation.PillowWriter(fps=fps)
    anim.save(f"Results/{folder}/animation.gif",writergif)
