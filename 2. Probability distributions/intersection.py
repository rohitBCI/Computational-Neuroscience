import scipy as sc
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def get_intersection_locations(y1,y2,test=False,x=None): 
    """
    return indices of the intersection point/s.
    """
    idxs=np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    if test:
        x=range(len(y1)) if x is None else x
        plt.figure(figsize=[5,5])
        ax=plt.subplot()
        ax.plot(x,y1,color='r',label='s1',alpha=0.5)
        ax.plot(x,y2,color='b',label='s2',alpha=0.5)
        _=[ax.axvline(x[i],color='k') for i in idxs]
        _=[ax.text(x[i],ax.get_ylim()[1],f"{x[i]:1.2f}",ha='center',va='bottom') for i in idxs]
        ax.legend(bbox_to_anchor=[1,1])
        ax.set(xlabel='x',ylabel='density')
        plt.show()
    return idxs

# single intersection
mu1 = 5
sigma1 = 0.5
x = np.linspace(mu1 - 4*sigma1, mu1 + 10*sigma1, 100)

mu2 = 7
sigma2 = 1

s1=sc.stats.norm.pdf(x,mu1,sigma1)
s2=sc.stats.norm.pdf(x,mu2,sigma2)
get_intersection_locations(y1=s1*2,y2=s2,x=x,test=True)