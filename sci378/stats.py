# import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas
from statsmodels.formula.api import ols

import os,sys
from numpy import nanmean
from scipy.stats import distributions as D
from numpy import *
from pylab import *
import pylab as py
import warnings


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


from .utils import Struct

from .best import histogram

from copy import deepcopy

greek=['alpha','beta','gamma','chi','tau','sigma','lambda',
        'epsilon','zeta','xi','theta','rho','psi','mu','nu','phi']

def remove_nan(x,y):
    try:
        x=x[y.notnull()]
        y=y[y.notnull()]
    except AttributeError:
        x=x[~isnan(y)]
        y=y[~isnan(y)]
        
    return x,y
    


    
def fit(x,y,funcstr,*args,**kwargs):

    x=pandas.Series(array(x))
    y=pandas.Series(array(y))

    x,y=remove_nan(x,y)
    
    
    if funcstr=='linear':
        result=fit(x,y,'power',1)
        result.type='linear'
    elif funcstr=='quadratic':
        result=fit(x,y,'power',2)
        result.type='quadratic'
    elif funcstr=='exponential':
        y2=np.log(y)
        result=fit(x,y2,'linear')
        result.params=[np.exp(result.params[1]),result.params[0]]
        p=result.params
        labelstr='y= %.4e exp(%.4e x)' % (p[0],p[1])
        result.label=labelstr
        result.type='exponential'
    
    elif funcstr=='power':
        data=pandas.DataFrame({'x':x,'y':y})
        power=args[0]
        
        keys=['x']
        for i in range(power-1):
            exponent=(i+2)
            key='x%d' % exponent
            data[key] = x**exponent
            keys.append(key)

        result2=sm.OLS(y=data['y'],x=data[keys])
        keys.reverse()
        keys+=['intercept']
        
        p=[result2.beta[s] for s in keys]

        labelstr='y= '
        for i,pv in enumerate(p):
            pw=len(p)-i-1
            if pw==1:
                labelstr+='%.4e x + ' % (pv)
            elif pw==0:
                labelstr+='%.4e + ' % (pv)
            else:
                labelstr+='%.4e x^%d + ' % (pv,pw)
        labelstr=labelstr[:-3]  # take off the last +
        
        
        result=Struct()
        result.params=p
        result.type='power'
        result.label=labelstr   
        result.pandas_result=result2
        
    else:
        raise ValueError('Unknown fit name %s' % funcstr)
        
    return result
        
def fitval(result,x):
    x=pandas.Series(array(x))

    if result.type=='linear':
        y=result.params[0]*x+result.params[1]
    elif result.type=='quadratic':
        y=result.params[0]*x**2+result.params[1]*x+result.params[2]
    elif result.type=='power':
        y=0.0
        for i,pv in enumerate(result.params):
            pw=len(result.params)-i-1
            y+=pv*x**pw
    elif result.type=='exponential':
        y=result.params[0]*np.exp(x*result.params[1])
    else:
        raise ValueError('Unknown fit name %s' % result.type)
        
    return y
    
      

try:
    import emcee
except ImportError:
    pass
    
def corner(samples,labels):
    N=len(labels)
    from matplotlib.colors import LogNorm
    
    py.figure(figsize=(12,12))
    
    axes={}
    for i,l1 in enumerate(labels):
        for j,l2 in enumerate(labels):
            if j>i:
                continue
                
            ax = py.subplot2grid((N,N),(i, j))
            axes[(i,j)]=ax
            
            idx_y=labels.index(l1)
            idx_x=labels.index(l2)
            x,y=samples[:,idx_x],samples[:,idx_y]
            
            if i==j:
                # plot distributions
                xx,yy=histogram(x,bins=200,plot=False)
                py.plot(xx,yy,'-o',markersize=3)
                py.gca().set_yticklabels([])
                
                if i==(N-1):
                    py.xlabel(l2)
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                else:
                    ax.set_xticklabels([])
                
            else:
                counts,ybins,xbins,image = py.hist2d(x,y,bins=100,norm=LogNorm())
                #py.contour(counts,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=3)
                
                if i==(N-1):
                    py.xlabel(l2)
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                else:
                    ax.set_xticklabels([])
                    
                if j==0:
                    py.ylabel(l1)
                    [l.set_rotation(45) for l in ax.get_yticklabels()]
                else:
                    ax.set_yticklabels([])

                ax.grid(True)
    
    # make all the x- and y-lims the same
    j=0
    lims=[0]*N
    for i in range(1,N):
        ax=axes[(i,0)]
        lims[i]=ax.get_ylim()

        if i==N-1:
            lims[0]=ax.get_xlim()
    
        
    for i,l1 in enumerate(labels):
        for j,l2 in enumerate(labels):
            if j>i:
                continue
                
            ax=axes[(i,j)]
            
            if j==i:
                ax.set_xlim(lims[i])
            else:
                ax.set_ylim(lims[i])
                ax.set_xlim(lims[j])



def normal(x,mu,sigma):
    return 1/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/2.0/sigma**2)

def num2str(a):
    from numpy import abs
    if a==0:
        sa=''
    elif 0.001<abs(a)<10000:
        sa='%g' % a
    else:
        sa='%.3e' % a
        parts=sa.split('e')

        parts[1]=parts[1].replace('+00','')
        parts[1]=parts[1].replace('+','')
        parts[1]=parts[1].replace('-0','-')
        parts[1]=parts[1].replace('-0','')
        
        sa=parts[0]+r'\cdot 10^{%s}'%parts[1]
    
    return sa
    

    
def linear_equation_string(a,b):
    
    astr=num2str(a)
    bstr=num2str(abs(b))
    
    if b<0:
        s=r'$y=%s\cdot x - %s$' % (astr,bstr)
    else:
        s=r'$y=%s\cdot x + %s$' % (astr,bstr)
    
    return s
    
def quadratic_equation_string(a,b,c):
    
    astr=num2str(a)
    bstr=num2str(abs(b))
    cstr=num2str(abs(c))
    
    s=r'$y=%s\cdot x^{2}' % astr
    
    
    if b<0:
        s+=r' - %s\cdot x' % (bstr)
    else:
        s+=r' - %s\cdot x' % (bstr)
    
    if c<0:
        s+=r' - %s$' % (cstr)
    else:
        s+=r' - %s$' % (cstr)

    return s



from scipy.special import gammaln,gamma

def logfact(N):
    return gammaln(N+1)

def tpdf(x,df,mu,sd):
    t=(x-mu)/float(sd)
    return gamma((df+1)/2.0)/sqrt(df*pi)/gamma(df/2.0)/sd*(1+t**2/df)**(-(df+1)/2.0)
    
def logtpdf(x,df,mu,sd):
    try:
        N=len(x)
    except TypeError:
        N=1
    
    t=(x-mu)/float(sd)
    return N*(gammaln((df+1)/2.0)-0.5*np.log(df*np.pi)-gammaln(df/2.0)-np.log(sd))+(-(df+1)/2.0)*np.sum(np.log(1+t**2/df))

def loguniformpdf(x,mn,mx):
    if mn < x < mx:
        return np.log(1.0/(mx-mn))
    return -np.inf

def logjeffreyspdf(x):
    
    if x>0.0:
        return -np.log(x).sum()
    return -np.inf

def logcauchypdf(x,x0,scale):
    return (-np.log(np.pi)-np.log(scale)-np.log(1 + ((x-x0)/scale)**2)).sum()

def loghalfcauchypdf(x,x0,scale):
    try:
        N=len(x)
    except TypeError:
        N=1

    if x<=0:
        return -np.inf

    return (-np.log(np.pi)-np.log(scale)-np.log(1 + ((x-x0)/scale)**2)).sum()

def loghalfnormalpdf(x,sig):
    # x>0: 2/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
    except TypeError:
        N=1
    if x<=0:
        return -np.inf
        
    return np.log(2)-0.5*np.log(2*np.pi*sig**2)*N - np.sum(x**2/sig**2/2.0)

def lognormalpdf(x,mn,sig,all_positive=False):
    # 1/sqrt(2*pi*sigma^2)*exp(-x^2/2/sigma^2)
    try:
        N=len(x)
        val=-0.5*np.log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)
        if all_positive:
            val[x<0]=-np.inf
        # print(x,mn,val)
        # raise ValueError("here")
        return val
    except TypeError:
        N=1
        # print(x,mn)
        # raise ValueError("there")
        val=-0.5*np.log(2*np.pi*sig**2)*N - np.sum((x-mn)**2/sig**2/2.0)
        if all_positive and x<0:
            val=-np.inf

        return val
         

def logexponpdf2(x,scale):
    if x<=0:
        return -np.inf
    return np.log(scale)-x/scale


def logbernoullipdf(theta, h, N):
    return logfact(N)-logfact(h)-logfact(N-h)+np.log(theta)*h+np.log(1-theta)*(N-h)

def logbetapdf(theta, h, N):
    if theta<0 or theta>1:
        return -np.inf
        
    return logfact(N+1)-logfact(h)-logfact(N-h)+np.log(theta)*h+np.log(1-theta)*(N-h)

def logexponpdf(x,_lambda):
    # p(x)=l exp(-l x)
    if x>0.0:
        return -_lambda*x + np.log(_lambda)
    return -np.inf

def logUniform(x,mn,mx):
    return loguniformpdf(x,mn,mx)

def logNormal(x,μ,σ):
    return lognormalpdf(x,μ,σ)

def logJeffreys(x):
    return logjeffreyspdf(x)

def logExponential(x,scale):
    value=logexponpdf2(x,scale)
    return value

def logStudent_T(x,df,μ,σ):
    value=logtpdf(x,df,μ,σ)
    return value


import scipy.optimize as op

class StudentT(object):
    def __init__(self,mean=0,std=1,dof=1):
        self.mean=mean
        self.std=std
        self.dof=dof
        self.default=mean
        self.D=D.t(self.dof,loc=self.mean,scale=self.std)

    def rand(self,*args):
        return np.random.randn(*args)*self.std+self.mean
    
    def __call__(self,x):
        return logtpdf(x,self.dof,self.mean,self.std)

class Normal(object):
    def __init__(self,mean=0,std=1,all_positive=False):
        self.mean=mean
        self.std=std
        self.default=mean
        self.all_positive=all_positive
        self.D=D.norm(self.mean,self.std)
        
    def rand(self,*args):

        return np.random.randn(*args)*self.std+self.mean
    
    def __call__(self,x):
        return lognormalpdf(x,self.mean,self.std,self.all_positive)

    def __str__(self):
        return "Normal(%g,%g)" % (self.mean,self.std)


class LogNormal(object):
    def __init__(self,mean=0,std=1):
        self.mean=mean
        self.std=std
        self.default=mean
        self.D=D.lognorm(self.mean,self.std)

    def rand(self,*args):
        return np.random.randn(*args)*self.std+self.mean
    
    def __call__(self,x):
        return loglognormalpdf(x,self.mean,self.std)


class Exponential(object):
    def __init__(self,_lambda=1):
        self._lambda=_lambda
        self.D=D.expon(self._lambda)

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logexponpdf(x,self._lambda)


class HalfCauchy(object):
    def __init__(self,x0=0,scale=1):
        self.x0=x0
        self.scale=scale
        self.default=x0
        self.D=D.halfcauchy(loc=self.x0,scale=self.scale) 

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return loghalfcauchypdf(x,self.x0,self.scale)


class HalfNormal(object):
    def __init__(self,sigma=1):
        self.sigma=sigma
        self.D=D.halfnorm(self.sigma)

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return loghalfnormalpdf(x,self.sigma)

class Uniform(object):
    def __init__(self,min=0,max=1):
        self.min=min
        self.max=max
        self.default=(min+max)/2.0
        self.D=D.uniform(self.min,self.max-self.min)

    def rand(self,*args):
        return np.random.rand(*args)*(self.max-self.min)+self.min
        
    def __call__(self,x):
        return loguniformpdf(x,self.min,self.max)

class Jeffreys(object):
    def __init__(self):
        self.default=1.0
        self.D=None # improper

    def rand(self,*args):
        return np.random.rand(*args)*2
        
    def __call__(self,x):
        return logjeffreyspdf(x)

class Cauchy(object):
    def __init__(self,x0=0,scale=1):
        self.x0=x0
        self.scale=scale
        self.default=x0
        self.D=D.cauchy(loc=self.x0,scale=self.scale) 

    def rand(self,*args):
        return np.random.rand(*args)*2-1
        
    def __call__(self,x):
        return logcauchypdf(x,self.x0,self.scale)


class Beta(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N
        a=self.h+1
        b=(self.N-self.h)+1
        self.D=D.beta(a,b)


    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbetapdf(x,self.h,self.N)

class Bernoulli(object):
    def __init__(self,h=100,N=100):
        self.h=h
        self.N=N
        self.default=float(h)/N
        self.D=D.bernoulli(self.default)

    def rand(self,*args):
        return np.random.rand(*args)
        
    def __call__(self,x):
        return logbernoullipdf(x,self.h,self.N)
     


class UniformLog(object):
    def __init__(self,min=0,max=1):
        self.min=min
        self.max=max
        self.default=np.exp((min+max)/2.0)
       
    def rand(self,*args):
        return np.exp(np.random.rand(*args)*(self.max-self.min)+self.min)
        
    def __call__(self,x):
        if x<=0.0:
            return -np.inf
        return loguniformpdf(log(x),self.min,self.max)



def lnprior_function(model):
    def _lnprior(x):
        return model.lnprior(x)

    return _lnprior


def dicttable(D):
    buf=[]
    add=buf.append
    add('<table>')
 
    for key in D:
        add('<tr><td><b>%s</b></td><td>%s</td></tr>' % (key,D[key]))
        
    add('</table>')
    return '\n'.join(buf) 

class MCMCModel_Meta(object):

    def __init__(self,**kwargs):
        self.params=kwargs
        self.warnings=[]
        
        self.keys=[]
        for key in self.params:
            self.keys.append(key)


        self.index={}
        for i,key in enumerate(self.keys):
            self.index[key]=i


        self.nwalkers=100
        self.burn_percentage=0.25
        self.initial_value=None
        self.samples=None
        self.last_pos=None
        self.max_iterator=1000  # for the sample iterator

        self.parallel = False

    def lnprior(self,theta):
        pass

    def lnlike(self,theta):
        pass

    def lnprob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)        

    def __call__(self,theta):
        return self.lnprob(theta)

    def set_initial_values(self,method='prior',verbose=False,*args,**kwargs):
        if method=='prior':
            ndim=len(self.params)
            try:
                N=args[0]
            except IndexError:
                N=300

            val=np.zeros(len(self.keys))
            pos=zeros((self.nwalkers,ndim))
            use_sample_ball=False
            for i,key in enumerate(self.keys):
                try:
                    pos[:,i]=self.params[key].rand(self.nwalkers)
                except AttributeError:
                    val[i]=self.params[key]
                    use_sample_ball=True

            if use_sample_ball:
                pos=emcee.utils.sample_ball(val, .05*val+1e-4, size=self.nwalkers)
                
            self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, 
                    lnprior_function(self))

            if verbose:
                timeit(reset=True)
                print("Sampling Prior...")
                
            with warnings.catch_warnings(record=True) as warning_list:
                self.sampler.run_mcmc(pos, N,**kwargs)

            self.warnings.extend(warning_list)
                

            if verbose:
                print("Done.")
                print((timeit()))

            # assign the median back into the simulation values
            self.burn()
            self.median_values=np.percentile(self.samples,50,axis=0)

            self.last_pos=self.sampler.chain[:,-1,:]
        elif method=='samples':
            lower,upper=np.percentile(self.samples, [16,84],axis=0)            
            subsamples=self.samples[((self.samples>=lower) & (self.samples<=upper)).all(axis=1),:]
            idx=np.random.randint(subsamples.shape[0],size=self.last_pos.shape[0])
            self.last_pos=subsamples[idx,:]            


        elif method=='maximum likelihood':
            self.set_initial_values()
            chi2 = lambda *args: -2 * self.lnlike_lownoise(*args)
            result = op.minimize(chi2, self.initial_value)
            vals=result['x']
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        elif method=='median':            
            vals=self.median_values
            self.last_pos=emcee.utils.sample_ball(vals, 
                            0.05*vals+1e-4, size=self.nwalkers)
        else:
            raise ValueError("Unknown method: %s" % method)

    def burn(self,burn_percentage=None):
        if not burn_percentage is None:
            self.burn_percentage=burn_percentage
            
        burnin = int(self.sampler.chain.shape[1]*self.burn_percentage)  # burn 25 percent
        ndim=len(self.params)
        self.samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    def run_mcmc(self,N,repeat=1,verbose=False,**kwargs):
        try:
            import multiprocess as mp
            mp.set_start_method('fork')            
        except (ImportError,RuntimeError,ValueError):
            self.parallel=False

        import os

        if self.parallel:
            os.environ["OMP_NUM_THREADS"] = "1"            


        ndim=len(self.params)
        
        if self.last_pos is None:
            self.set_initial_values()
        
        
        for i in range(repeat):

            timeit(reset=True)

            if not self.parallel:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self,)

                if verbose:
                    if repeat==1:
                        print("Running MCMC...")
                    else:
                        print("Running MCMC %d/%d..." % (i+1,repeat))

                with warnings.catch_warnings(record=True) as warning_list:
                    self.sampler.run_mcmc(self.last_pos, N,**kwargs)

                self.warnings.extend(warning_list)
                        
            else:
                with mp.Pool() as pool:
                    self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self, pool=pool)
                    if repeat==1:
                        print("Running Parallel MCMC...")
                    else:
                        print("Running Parallel MCMC %d/%d..." % (i+1,repeat))

                    with warnings.catch_warnings(record=True) as warning_list:
                        self.sampler.run_mcmc(self.last_pos, N,**kwargs)
            
                    self.warnings.extend(warning_list)
                        
                    

            if verbose:
                print("Done.")
                print((timeit()))


            # assign the median back into the simulation values
            self.burn()
            self.median_values=np.percentile(self.samples,50,axis=0)
            theta=self.median_values


            self.last_pos=self.sampler.chain[:,-1,:]
            if repeat>1:
                self.set_initial_values('samples')  # reset using the 16-84 percentile values from the samples


    def plot_chains(self,*args,**kwargs):
        py.clf()
        
        if not args:
            args=self.keys
        
        
        fig, axes = py.subplots(len(self.params), 1, sharex=True, figsize=(8, 5*len(args)))
        try:  # is it iterable?
            axes[0]
        except TypeError:
            axes=[axes]



        labels=[]
        for ax,key in zip(axes,args):
            i=self.index[key]
            sample=self.sampler.chain[:, :, i].T

            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='$%s$' % namestr

            labels.append(label)
            ax.plot(sample, color="k", alpha=0.2,**kwargs)
            ax.set_ylabel(label)

    def triangle_plot(self,*args,**kwargs):
        
        if not args:
            args=self.keys
            
        assert len(args)>1
        
        labels=[]
        idx=[]
        for key in args:
            if key.startswith('_sigma'):
                label=r'$\sigma$'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='$%s$' % namestr

            labels.append(label)
            idx.append(self.index[key])
        
        fig = corner(self.samples[:,idx], labels=labels,**kwargs)

            
    def plot_distributions(self,*args,kde=False,xlim=None,**kwargs):
        from scipy.stats import gaussian_kde
        from scipy.stats import distributions as D


        if not args:
            args=self.keys
        
        for key in args:
            if key.startswith('_sigma'):
                label=r'\sigma'
            else:
                namestr=key
                for g in greek:
                    if key.startswith(g):
                        namestr=r'\%s' % key

                label='%s' % namestr

            
            py.figure(figsize=(12,4))
            samples=self.get_samples(key)
            
            x,y=histogram(samples,bins=200,plot=False)
            if xlim is None:
                xl=(x[0],x[-1])
            else:
                xl=xlim.get(key,(x[0],x[-1]))
                
            
            
            if kde:
                density = gaussian_kde(samples.ravel())
                xx=linspace(xl[0],xl[1],200)
                yy=density(xx)
                plot(xx,yy,lw=2)
                fill_between(xx,yy,facecolor='blue', alpha=0.2)

                HDI=np.percentile(samples.ravel(), [2.5, 50, 97.5],axis=0)
                yl=gca().get_ylim()
                text((HDI[0]+HDI[2])/2, 0.15*yl[1],'95% HDI', ha='center', va='center',fontsize=12)
                plot(HDI,[yl[1]*.1,yl[1]*.1,yl[1]*.1],'k.-',linewidth=1)
                for v in HDI:
                    if v<0.005:
                        text(v, 0.05*yl[1],'%.3g' % v, ha='center', va='center', 
                             fontsize=12)
                    else:
                        text(v, 0.05*yl[1],'%.3f' % v, ha='center', va='center', 
                             fontsize=12)
                
                
            else:
                result=histogram(samples,bins=200,plot=True)
                v=np.percentile(samples, [2.5, 50, 97.5],axis=0)
                py.title(r'$\hat{%s}^{97.5}_{2.5}=%.3f^{%.3f}_{%.3f}$' % (label,v[1],v[2],v[0]))
            
            
            gca().set_xlim(*xl)

            py.ylabel(r'$p(%s|{\rm data})$' % label)
            py.xlabel(r'$%s$' % label)
                
    def get_distribution(self,key,bins=200):
            
        if key in self.index:
            i=self.index[key]
            x,y=histogram(self.samples[:,i],bins=bins,plot=False)
        else:
            x,y=histogram(self.get_samples(key)[0],bins=bins,plot=False)
        
        return x,y
        
    def percentiles(self,p=[16, 50, 84],S=None):
        result={}


        if S is None:
            for i,key in enumerate(self.keys):
                result[key]=np.percentile(self.samples[:,i], p,axis=0)
        else:    
            result[S]=np.percentile(self.get_samples(S),p,axis=0)

        return result

        
    def best_estimates(self,S=None):

        self.median_values=np.percentile(self.samples,50,axis=0)
        theta=self.median_values
        
        return self.percentiles(S=S)

    def sample_iterator(self,*args):
        s=self.get_samples(*args)
        L=min(s.shape[0],self.max_iterator)
        for i in range(L):
            yield s[i,:]

    def random_sample(self):
        return choice(self.samples)


    def get_samples(self,*args):
        from numpy import sqrt,log,sin,cos,tan,exp,array
        import numpy as np

        if not args:
            args=self.keys
        
        result=[]
        for arg in args:
            if arg in self.keys:
                idx=self.keys.index(arg)
                result.append(self.samples[:,idx])
            else:
                D={}
                for key in self.keys:
                    v=array(self.get_samples(key))
                    D[key]=v
                
                D['np']=np
                for name,fun in zip(['sqrt','log','sin','cos','tan','exp','array'],
                                    [sqrt,log,sin,cos,tan,exp,array]):
                    D[name]=fun
                
                N=float(np.prod(v.shape))
                D['N']=N
            
                result.append(eval(arg,D).squeeze())

        return np.atleast_2d(array(result)).T
    
    def BIC(self):
        L=self.lnlike(self.median_values)
        return log(self.N)*self.k- 2*L

    def WAIC(self):
        # from https://github.com/pymc-devs/pymc3/blob/02f0b7f9a487cf18e9a48b754b54c2a99cf9fba8/pymc3/stats.py
        # We get three different measurements:
        # waic: widely available information criterion
        # waic_se: standard error of waic
        # p_waic: effective number parameters

        from scipy.special import logsumexp
        log_py=np.atleast_2d(array([self.lnprob(theta) 
                                        for theta in self.samples])).T
        lppd_i = logsumexp(log_py, axis=0, b=1.0 / len(log_py))
        vars_lpd = np.var(log_py, axis=0)
        warn_mg = 0
        if np.any(vars_lpd > 0.4):
            warnings.warn("""For one or more samples the posterior variance of the
            log predictive densities exceeds 0.4. This could be indication of
            WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
            """)
            warn_mg = 1

        waic_i = - 2 * (lppd_i - vars_lpd)
        waic = np.sum(waic_i)
        waic_se = np.sqrt(len(waic_i) * np.var(waic_i))
        p_waic = np.sum(vars_lpd)            

        return waic,waic_se,p_waic

    def BayesFactor(self,r=0.05):
        # from http://www.astroml.org/book_figures/chapter5/fig_model_comparison_mcmc.html
        from scipy.special import gamma
        import numpy as np
        from sklearn.neighbors import BallTree

        traces=self.samples
        logp=np.array([self.lnprob(s) for s in self.samples])

        """Estimate the bayes factor using the local density of points"""
        N, D = traces.shape

        # compute volume of a D-dimensional sphere of radius r
        Vr = np.pi ** (0.5 * D) / gamma(0.5 * D + 1) * (r ** D)

        # use neighbor count within r as a density estimator
        bt = BallTree(traces)
        count = bt.query_radius(traces, r=r, count_only=True)

        self.BF = logp + np.log(N) + np.log(Vr) - np.log(count)

        p25, p50, p75 = np.percentile(self.BF, [25, 50, 75])
        return p50, 0.7413 * (p75 - p25)

    def estimate_bayes_factor(traces, logp, r=0.05, return_list=False):
        from scipy.special import gamma
        import numpy as np
        from sklearn.neighbors import BallTree
        
        """Estimate the bayes factor using the local density of points"""
        N, D = traces.shape

        # compute volume of a D-dimensional sphere of radius r
        Vr = np.pi ** (0.5 * D) / gamma(0.5 * D + 1) * (r ** D)

        # use neighbor count within r as a density estimator
        bt = BallTree(traces)
        count = bt.query_radius(traces, r=r, count_only=True)

        BF = logp + np.log(N) + np.log(Vr) - np.log(count)

        if return_list:
            return BF
        else:
            p25, p50, p75 = np.percentile(BF, [25, 50, 75])
            return p50, 0.7413 * (p75 - p25)

    def P(self,S):
        from numpy import sqrt,log,sin,cos,tan,exp,array
        import numpy as np

        D={}
        for key in self.keys:
            v=array(self.get_samples(key))
            D[key]=v
        
        D['np']=np
        for name,fun in zip(['sqrt','log','sin','cos','tan','exp','array'],
                            [sqrt,log,sin,cos,tan,exp,array]):
            D[name]=fun
        
        N=float(np.prod(v.shape))
        D['N']=N
        
        result=eval('np.sum(%s)/N' % S,D)
        return result



    def summary(self):
        buf=[]
        add=buf.append
        
        def tds(L):
            return ' '.join([f'<td>{s}</td>' for s in L])
        def b(s):
            s=str(s)
            return f"<b>{s}</b"
        
        add('<h1>Priors</h1>')
        add('<table>')
        
        row=[]
        td=row.append
        
        td(b("name"))
        td(b("prior"))
        td(b(" "))
        
        add('<tr>%s</tr>' % (tds(row)))
        for p in self.params:
            typestr=str(type(self.params[p])).split("'")[1].replace('pyndamics3.mcmc.','')
            
            row=[]
            td=row.append
            
            td(b(p))
            td(typestr)
        
            if typestr=='Uniform':
                td(dicttable(
                    {'min':self.params[p].min,
                    'max':self.params[p].max}))
            else:
                td(dicttable({}))
                
            add('<tr>%s</tr>' % (tds(row)))
        
        add('</table>')
        
        add('<h1>Fit Statistics</h1>')
            
        N=sum([len(c.data['value']) for c in model.sim.components if c.data])
        fit_stats={'data points':N,
                'variables':len(model.params),
                'number of walkers':model.nwalkers,
                'number of samples':model.samples.shape[0],
                'Bayesian info crit. (BIC)':model.BIC}
        add(dicttable(fit_stats))
        
            
        
        add('<h1>Posteriors</h1>')
        add('<table>')
        
        row=[]
        td=row.append
        
        td(b("name"))
        td(b("value"))
        td(b("2.5%"))
        td(b("97.5%"))
        
        add('<tr>%s</tr>' % (tds(row)))
        
        pp=self.percentiles([0.025,50,97.5])
        for p in self.params:
            typestr=str(type(self.params[p])).split("'")[1].replace('pyndamics3.mcmc.','')
            
            row=[]
            td=row.append
            
            td(b(p))
            td(f'{pp[p][1]:.5g}')
            td(f'{pp[p][0]:.5g}')
            td(f'{pp[p][2]:.5g}')

            add('<tr>%s</tr>' % (tds(row)))
        
        add('</table>')
            
        
        
        
        #return HTML('\n'.join(buf))
        return '\n'.join(buf)        





class MCMCModel(MCMCModel_Meta):
    def __init__(self,data,lnlike,lnprior=None,prior_kwargs={},**kwargs):
        import inspect

        self.data=data
        self.lnprior_function=lnprior
        self.lnlike_function=lnlike
        self.prior_kwargs=prior_kwargs

        # if not lnprior is None:

        #     func=lnprior

        #     pos_args = []
        #     kw_args = {}
        #     keywords_ = None
        #     sig = inspect.signature(func)
        #     for fnam, fpar in sig.parameters.items():
        #         if fpar.kind == fpar.VAR_KEYWORD:
        #             keywords_ = fnam
        #         elif fpar.kind == fpar.POSITIONAL_OR_KEYWORD:
        #             if fpar.default == fpar.empty:
        #                 pos_args.append(fnam)
        #             else:
        #                 kw_args[fnam] = fpar.default
        #         elif fpar.kind == fpar.VAR_POSITIONAL:
        #             raise ValueError("varargs '*%s' is not supported" % fnam)
        #     # inspection done

        #     kwargs={k:None for k in pos_args}

        MCMCModel_Meta.__init__(self,**kwargs)

        self.k=len(self.params)
        self.N=len(self.data)


    def lnprior(self,theta):
        if self.lnprior_function is None:
            value=0.0
            for i,key in enumerate(self.keys):
                value+=self.params[key](theta[i])
                    
            return value
        else:
            params_dict={}
            for i,key in enumerate(self.keys):
                params_dict[key]=theta[i]
                    
            D=dict(params_dict, **self.prior_kwargs)
            return self.lnprior_function(**D)


    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            params_dict[key]=theta[i]
        return self.lnlike_function(self.data,**params_dict)




class MCMCModel1(MCMCModel_Meta):
    
    def __init__(self,x,y,function,**kwargs):
        self.x=x
        self.y=y
        self.function=function
        self.params=kwargs
        
        MCMCModel_Meta.__init__(self,**kwargs)

        self.params['_sigma']=Jeffreys()
        self.keys.append('_sigma')        
        self.index['_sigma']=len(self.keys)-1

        self.k=len(self.params)
        self.N=len(self.x)


    # Define the probability function as likelihood * prior.
    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=theta[i]
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)
    
    def lnlike_lownoise(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            if key=='_sigma':
                sigma=1.0
            else:
                params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,sigma)


    def predict(self,x,theta=None):
        
        if theta is None:
            self.percentiles()
            theta=self.median_values
            
        args={}
        for v,key in zip(theta,self.keys):
            if key=='_sigma':
                continue
            args[key]=v

        y_predict=array([self.function(_,**args) for _ in x])        
        return y_predict
    
    def plot_predictions(self,x,N=1000,color='k'):
        samples=self.samples[-N:,:]

        predictions=[]
        for value in samples:
            args={}
            for v,key in zip(value,self.keys):
                if key=='_sigma':
                    continue
                args[key]=v

            y_predict=array([self.function(_,**args) for _ in x])        
            plot(x,y_predict,color=color,alpha=0.05)
            predictions.append(y_predict)

        return predictions

class MCMCModelErr(MCMCModel):

    def __init__(self,x,y,yerr,function,**kwargs):
        self.x=x
        self.y=y
        self.yerr=yerr
        self.function=function
        self.params=kwargs
        
        MCMCModel_Meta.__init__(self,**kwargs)

        self.k=len(self.params)
        self.N=len(self.x)


    # Define the probability function as likelihood * prior.
    def lnprior(self,theta):
        value=0.0
        for i,key in enumerate(self.keys):
            value+=self.params[key](theta[i])
                
        return value

    def lnlike(self,theta):
        params_dict={}
        for i,key in enumerate(self.keys):
            params_dict[key]=theta[i]
                
        y_fit=self.function(self.x,**params_dict)
        
        return lognormalpdf(self.y,y_fit,self.yerr)

    
    
def plot_many(model,lmfit_model,x,color='b',N=500):
    samples=model.get_samples()
    params=lmfit_model.make_params()
    D={'params':params,
      lmfit_model.independent_vars[0]:x}
    for i in range(N):
        s=np.random.randint(samples.shape[0])
        theta=samples[s,:]


        for i,key in enumerate(model.params):
            if key in params:
                params[key].value=theta[i]

        
        y=lmfit_model.eval(**D)
        
        plot(x,y,'-',color=color,alpha=0.01)
        
    median=model.percentiles(50)
    for i,key in enumerate(model.params):
        if key in params:
            params[key].value=median[key]
        
    y=lmfit_model.eval(**D)
    plot(x,y,'-',color=color)
    