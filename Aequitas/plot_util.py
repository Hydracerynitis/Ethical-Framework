import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_scatter(ax,x,y,title,annotations):
    colors = cm.rainbow(np.linspace(0, 1, len(x)))
    ax.scatter(x, y,c=colors)
    y_range=max(y)-min(y)
    x_range=max(x)-min(x)
    for i in range(len(x)):
        ax.text(x[i]-x_range*0.15,y[i]+y_range*0.02,annotations[i]) 
    ax.set(xlim=(min(x)-x_range*0.3, max(x)+x_range*0.3))
    ax.axline((0, 1.05), (1, 1.05), linewidth=2, color='b')
    ax.axline((0, 0.95), (1, 0.95), linewidth=2, color='b')
    ax.axline((0, 1), (1, 1), linewidth=2, color='r')
    ax.text(min(x)-x_range*0.3,1+y_range*0.02,"Ideal Performance")
    ax.text(min(x)-x_range*0.3,1.05+y_range*0.02,"Significance Level")
    ax.text(min(x)-x_range*0.3,0.95+y_range*0.02,"Significance Level")
    ax.set_title(title)