from matplotlib import rcParams
from numpy import array

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols,wls


fontsize=20

rcParams['font.size']=fontsize
rcParams['font.family']='sans-serif'

rcParams['axes.labelsize']=fontsize
rcParams['axes.titlesize']=fontsize
rcParams['xtick.labelsize']=fontsize
rcParams['ytick.labelsize']=fontsize
rcParams['legend.fontsize']=fontsize

rcParams['figure.figsize']=(12,8)

rcParams['axes.grid']=True

def xaxis_dates():
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()


def plot_correlations(data,*args):
    if len(args)>0:
        data=data[list(args)]
        
    rho = data.corr()
    L=len(data.columns)

    pval = rho.copy()
    rho1= rho.copy()        
        
    from scipy.stats import pearsonr
    from numpy import zeros
    import pylab as plt
    from pylab import cm
    from matplotlib.patches import Ellipse
    
    for i,ci in enumerate(data.columns): # rows are the number of rows in the matrix.
        for j,cj in enumerate(data.columns):
            subdata=data[[ci,cj]]
            subdata=subdata.dropna()
            x=subdata.iloc[:,0]
            y=subdata.iloc[:,1]
            r,p=pearsonr(x,y)

            rho1[ci][cj]=r
            pval[ci][cj]  = p
        
        
    C=rho1
    
    fig=plt.gcf()
    # not sure what colormap to use
    # check out https://matplotlib.org/examples/color/colormaps_reference.html
    colors=cm.get_cmap('seismic')

    ax = fig.add_subplot(111, aspect='equal')
    cols=list(C.columns)
    L=len(cols)
    for i,iname in enumerate(cols):
        for j,jname in enumerate(cols):
            x=j
            y=L-i-1  # y-values are reversed of rows
            c=C.values[i,j]
            if c>0:
                ang=45
            else:
                ang=135

            if i>j:
                e=Ellipse(xy=(x,y),width=1,height=1-abs(c),angle=ang)
                e.set_facecolor(colors((-c+1)/2))  # convert -1..1 to 0..1 for color

                # red for positive correlation
                #e.set_facecolor(colors((c+1)/2))  # convert -1..1 to 0..1 for color

                e.set_edgecolor('black')
                ax.add_artist(e)
            elif i<j:
                if pval[iname][jname]<0.001:
                    sig=r'***'
                elif pval[iname][jname]<0.01:
                    sig='**'
                elif pval[iname][jname]<0.05:
                    sig='*'
                else:
                    sig=''
                plt.text(x,y,'%.3f' % (c),ha='center',va='center',fontsize=16)
                if sig:
                    plt.text(x,y+0.25,sig,ha='center',va='center',fontsize=13)
            else:
                continue

    ax.set_xlim(-1, L)
    ax.set_ylim(-1, L)        

    ax.set_xticks(range(L))
    ax.xaxis.tick_top()
    ax.set_xticklabels(cols,rotation=90)

    ax.set_yticks(range(L))
    cols.reverse()# y-values are reversed of rows
    ax.set_yticklabels(cols)

    ax.grid(False)

    import matplotlib 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)     
            
            
    return rho1,pval
