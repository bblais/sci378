def get_course_github(name='Computer-Programming-For-the-Sciences-Spring-2021',
                     folder=None):
    import os,sys,platform
    from pathlib import Path
    import IPython


    _ip=IPython.get_ipython()


    github_url='https://github.com/bblais/'+name
    
    if folder is None:
        downloads_folder=str(Path.home() / "Downloads")
    else:
        downloads_folder=folder
                         
    if not os.path.exists(downloads_folder):
        raise ValueError("Can't find folder %s" % downloads_folder)
        
    local_folder=str(Path(downloads_folder)  / name)
    
    print(github_url," ---> ",local_folder)
    
    
    if not os.path.exists(local_folder):
        if platform.system()=="Windows":
            print(_ip.getoutput(f"cd {downloads_folder} & git clone {github_url}"))
        else:
            print(_ip.getoutput(f"cd {downloads_folder} ; git clone {github_url}"))
            
    else:
        if platform.system()=="Windows":
            print(_ip.getoutput(f"cd {local_folder} & git pull {github_url}"))
        else:
            print(_ip.getoutput(f"cd {local_folder} ; git pull {github_url}"))

class Struct(dict):
    
    def __getattr__(self,name):
        
        try:
            val=self[name]
        except KeyError:
            val=super(Struct,self).__getattribute__(name)
            
        return val
    
    def __setattr__(self,name,val):
        
        self[name]=val



class Storage(object):
    def __init__(self):
        self.data=[]
    
    def __add__(self,other):
        s=Storage()
        s+=other
        return s
        
    def __iadd__(self,other):
        self.append(*other)
        return self
        
    def append(self,*args):
        if not self.data:
            for arg in args:
                self.data.append([arg])

        else:
            for d,a in zip(self.data,args):
                d.append(a)
       
    def DataFrame(self,variable_names=None):
        from pandas import DataFrame

        if variable_names is None:
            keys=['Variable %d' % _ for _ in range(len(self.data))]
        else:
            keys=variable_names

        return DataFrame({k:v for k,v in zip(keys,self.data)})


    def arrays(self):
        from numpy import array

        for i in range(len(self.data)):
            self.data[i]=array(self.data[i])

        ret=tuple(self.data)
        if len(ret)==1:
            return ret[0]
        else:
            return ret

    def __array__(self):
        from numpy import vstack
        return vstack(self.arrays())


    
def ols_result_random_samples(results,N=1000):
    from copy import deepcopy
    results2=deepcopy(results)
    
    r=np.random.multivariate_normal(np.array(results.params),
                                2*results.cov_HC0,N)
    
    for p in r:
        results2.params[:]=p
            
        yield results2


def date_to_float(d):
    
    from dateutil import parser
    import datetime  
    from numpy import array
    
    try:
        dt=parser.parse(d)
        year=dt.year
        f=year+(dt-datetime.datetime(year, 1, 1, 0, 0))/(datetime.datetime(year+1, 1, 1, 0, 0)-datetime.datetime(year, 1, 1, 0, 0))

        return f
    except TypeError:
        f=[date_to_float(_) for _ in array(d)]
        return array(f)
