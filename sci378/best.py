
# coding: utf-8

# ## Implementation of Kruschke's BEST Test
# 
# * Paper at http://www.indiana.edu/~kruschke/articles/Kruschke2013JEPG.pdf
# * Supporting website at http://www.indiana.edu/~kruschke/BEST/

# In[1]:


# In[2]:

import pylab as py
import emcee

from numpy import array,log
import numpy as np


# ## Style for plotting

# In[3]:

style_text="""
axes.titlesize : 24
axes.labelsize : 20
lines.linewidth : 3
lines.markersize : 5
xtick.labelsize : 20
ytick.labelsize : 20
figure.figsize : 10,8
axes.grid: True
grid.linestyle: -
grid.color: 0.75
font.size: 20
font.family: sans-serif
legend.fontsize: 20
legend.frameon: False
legend.numpoints: 1
legend.scatterpoints: 1
lines.solid_capstyle: round

text.color: .15
xtick.color: .15
ytick.color: .15
xtick.direction: out
ytick.direction: out

axes.axisbelow: True

# these are tableau20 colors
axes.prop_cycle : cycler('color', ['1f77b4', 'aec7e8','ff7f0e','ffbb78','2ca02c','98df8a','d62728','ff9896','9467bd','c5b0d5','8c564b','c49c94','e377c2','f7b6d2','7f7f7f','c7c7c7','bcbd22','dbdb8d','17becf','9edae5'])
"""

with open('mystyle.mplstyle','w') as fid:
    fid.write(style_text)
    
py.style.use('mystyle.mplstyle')    


# In[4]:

def histogram(y,bins=50,plot=True):
    N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    if dx==0.0:  #  all in 1 bin!
        val=bins[0]
        bins=np.linspace(val-abs(val),val+abs(val),50)
        N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    x=bins[0:-1]+(bins[1]-bins[0])/2.0
    
    y=N*1.0/np.sum(N)/dx
    
    if plot:
        py.plot(x,y,'o-')
        yl=py.gca().get_ylim()
        py.gca().set_ylim([0,yl[1]])
        xl=py.gca().get_xlim()
        if xl[0]<=0 and xl[0]>=0:    
            py.plot([0,0],[0,yl[1]],'k--')

    return x,y



# ## Implementation of the BEST model and support code

# In[5]:

import time

def time2str(tm):
    
    frac=tm-int(tm)
    tm=int(tm)
    
    s=''
    sc=tm % 60
    tm=tm//60
    
    mn=tm % 60
    tm=tm//60
    
    hr=tm % 24
    tm=tm//24
    dy=tm

    if (dy>0):
        s=s+"%d d, " % dy

    if (hr>0):
        s=s+"%d h, " % hr

    if (mn>0):
        s=s+"%d m, " % mn


    s=s+"%.2f s" % (sc+frac)

    return s

def timeit(reset=False):
    global _timeit_data
    try:
        _timeit_data
    except NameError:
        _timeit_data=time.time()

    if reset:
        _timeit_data=time.time()

    else:
        return time2str(time.time()-_timeit_data)


# In[6]:

from scipy.special import gammaln,gamma

def tpdf(x,df,mu,sd):
    t=(x-mu)/float(sd)
    return gamma((df+1)/2.0)/sqrt(df*pi)/gamma(df/2.0)/sd*(1+t**2/df)**(-(df+1)/2.0)
    
def logtpdf(x,df,mu,sd):
    try:
        N=len(x)
    except TypeError:
        N=1
    
    t=(x-mu)/float(sd)
    return N*(gammaln((df+1)/2.0)-0.5*log(df*np.pi)-gammaln(df/2.0)-np.log(sd))+(-(df+1)/2.0)*np.sum(np.log(1+t**2/df))

def loguniformpdf(x,mn,mx):
    if mn < x < mx:
        return np.log(1.0/(mx-mn))
    return -np.inf

def logjeffreyspdf(x):
    if x>0.0:
        return -np.log(x)
    return -np.inf

def lognormalpdf(x,mn,sig):
    # 1/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
        
    return -0.5*log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)

def logexponpdf(x,scale):
    if x<=0:
        return -np.inf
    return -np.log(scale)-x/scale

class BESTModel(object):
    
    def __init__(self,y1,y2):
        self.data=array(y1,float),array(y2,float)
        pooled=np.concatenate((y1,y2))
        self.S=np.std(pooled)
        self.M=np.mean(pooled)
        
        self.names=['μ1','μ2','σ1','σ2','ν']
        self.params=[]
        self.params_dict={}
        
    def initial_value(self):
        return np.array([self.M,self.M,self.S,self.S,10])
    
    def prior(self,theta):
        μ1,μ2,σ1,σ2,ν=theta
        value=0.0
        value+=lognormalpdf(μ1,self.M,1000*self.S)
        value+=lognormalpdf(μ2,self.M,1000*self.S)
        
        mn=0.001*self.S
        mx=1000*self.S
        value+=loguniformpdf(σ1,mn,mx-mn)
        value+=loguniformpdf(σ2,mn,mx-mn)

        value+=logexponpdf(ν-1,scale=29)
        return value
        
    def run_mcmc(self,iterations=1000,burn=0.1):
        # Set up the sampler.
        ndim, nwalkers = len(self), 100
        val=self.initial_value()
        pos=emcee.utils.sample_ball(val, .05*val+1e-4, size=nwalkers)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self)
        
        timeit(reset=True)
        print("Running MCMC...")
        self.sampler.run_mcmc(pos, iterations)
        print("Done.")
        print( timeit())
        
        burnin = int(self.sampler.chain.shape[1]*burn)
        samples = self.sampler.chain[:, burnin:, :]
        self.μ1=samples[:,:,0]
        self.μ2=samples[:,:,1]
        self.σ1=samples[:,:,2]
        self.σ2=samples[:,:,3]
        self.ν=samples[:,:,4]
        
        self.params=[self.μ1,self.μ2,self.σ1,self.σ2,self.ν]
        self.params_dict['μ1']=self.μ1
        self.params_dict['μ2']=self.μ2
        self.params_dict['σ1']=self.σ1
        self.params_dict['σ2']=self.σ2
        self.params_dict['ν']=self.ν
        
        
        
    def __len__(self):
        return 5  # μ1,μ2,σ1,σ2,ν
        
    def likelihood(self,theta):
        μ1,μ2,σ1,σ2,ν=theta
        y1,y2=self.data
        
        value=0.0
        value+=logtpdf(y1,ν,μ1,σ1)
        value+=logtpdf(y2,ν,μ2,σ2)

        return value
    
    def plot_chains(self,S,*args,**kwargs):
        from numpy import sqrt,log,sin,cos,tan,exp

        μ1,μ2,σ1,σ2,ν=self.params
        N=float(np.prod(μ1.shape))

        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='μ1':
                name=r"\mu_1"
            elif name=='μ2':
                name=r"\mu_2"
            elif name=='σ1':
                name=r"\sigma_1"
            elif name=='σ2':
                name=r"\sigma_2"
            elif name=='ν':
                name=r"\nu"
            else:
                name=r"%s" % name
        
            value=eval(S)
        
        py.plot(value, color="k",alpha=0.02,**kwargs)
        if "\\" in name:        
            py.ylabel("$"+name+"$")        
        else:
            py.ylabel(name)        
    
    def plot_distribution(self,S,p=95):
        from numpy import sqrt,log,sin,cos,tan,exp


        pp=[(100-p)/2.0,50,100-(100-p)/2.0]
        
        μ1,μ2,σ1,σ2,ν=self.params
        N=float(np.prod(μ1.shape))
        
        if '=' in S:
            name,expr=S.split('=')
            value=eval(expr)
        else:
            name=S
            if name=='μ1':
                name=r"\hat{\mu}_1"
            elif name=='μ2':
                name=r"\hat{\mu}_2"
            elif name=='σ1':
                name=r"\hat{\sigma}_1"
            elif name=='σ2':
                name=r"\hat{\sigma}_2"
            elif name=='ν':
                name=r"\hat{\nu}"
            else:
                name=r"\hat{%s}" % name
        
            value=eval(S)
            
        result=histogram(value.ravel(),bins=200)
        v=np.percentile(value.ravel(), pp ,axis=0)
        if r"\hat" in name:
            py.title(r'$%s=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        else:
            py.title(r'%s$=%.3f^{+%.3f}_{-%.3f}$' % (name,v[1],(v[2]-v[1]),(v[1]-v[0])),va='bottom')
        
        
    def P(self,S):
        from numpy import sqrt,log,sin,cos,tan,exp
        
        μ1,μ2,σ1,σ2,ν=self.params
        N=float(np.prod(μ1.shape))
        result=eval('np.sum(%s)/N' % S)
        return result
            
    def posterior(self,theta):
        prior = self.prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior + self.likelihood(theta)
        
    def __call__(self,theta):
        return self.posterior(theta)

