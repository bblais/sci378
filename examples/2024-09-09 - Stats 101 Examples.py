#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from sci378.stats import *


# ## Best Estimate, $\mu$, knowing $\sigma$

# In[ ]:


def logprior(μ):
    value=0
    
    value+=logNormal(μ,0,100)
    
    return value

def loglikelihood(data,μ):
    x=data
    
    value=0
        
    value+=logNormal(x-μ,0,σ)
    return value
    


# In[ ]:


data=array([12.0,14,16])
σ=1


# In[ ]:


model=MCMCModel(data,loglikelihood,logprior)   


# In[ ]:


model.run_mcmc(800,repeat=3,verbose=True)
model.plot_chains()


# In[ ]:


model.plot_distributions()


# In[ ]:


model.P("μ>15")


# ## Best Estimate, $\mu$, not knowing $\sigma$

# In[ ]:


def logprior(μ,σ):
    value=0
    
    value+=logNormal(μ,0,100)
    value+=logJeffreys(σ)
    
    return value

def loglikelihood(data,μ,σ):
    x=data
    
    value=0
        
    value+=logNormal(x-μ,0,σ)
    return value
    


# In[ ]:


model=MCMCModel(data,loglikelihood,logprior)   
model.run_mcmc(800,repeat=3,verbose=True)
model.plot_chains()


# In[ ]:


model.plot_distributions()


# In[ ]:


model.P("μ>15")


# In[ ]:


model.P("σ<1")


# ## Proportion

# In[ ]:


def logprior(θ):
    value=0
    
    value+=logNormal(θ,0,1)
    
    return value

def loglikelihood(data,θ):
    h,N=data
    
    value=0
        
    value+=logBernoulli(θ,h,N)
    return value
    


# In[ ]:


model=MCMCModel(data,loglikelihood,logprior)   
model.run_mcmc(800,repeat=3,verbose=True)
model.plot_chains()


# In[ ]:


model.plot_distributions()


# In[ ]:


model.P("θ<0.5")

