import numpy as np
#from scipy.integrate import odeint as odeint_np
#from jax.experimental.ode import odeint as odeint_jnp
import jax.numpy as jnp
#from jax import jit
#from jax.random import PRNGKey
import re
from functools import partial


class Reaction:
    
    @staticmethod
    def get_net_stoich(react,prod):
        net_s=prod.copy()
        for k,v in react.items():
            if k in net_s:
                net_s[k]=net_s[k]-v
            else:
                net_s[k]=-v
        return net_s
    
    @staticmethod
    def key_and_stoch_from_string(component):
        res=re.match("^([0-9]*)(\w*)",component)
        if res is None:
            return None
        g=res.groups()
        if len(g)!=2 or len(g[1])==0:
            return None
        stoch=1 if len(g[0])==0 else int(g[0])
        return (g[1],stoch)
    
    @staticmethod
    def get_stoich_from_string(a_string_sum):
        comp=a_string_sum.strip().split('+')
        s_dict={}
        for c in comp:
            ks=Reaction.key_and_stoch_from_string(c)
            if ks:
                s_dict[ks[0]]=ks[1]
        return s_dict

    
    @staticmethod
    def string_to_reaction(astring):
        react_prod=astring.split(">")
        if len(react_prod)!=2:
            return None
        r_dict=Reaction.get_stoich_from_string(react_prod[0])
        p_dict=Reaction.get_stoich_from_string(react_prod[1])
        net_dict=Reaction.get_net_stoich(r_dict,p_dict)
        return r_dict,net_dict       
        
    def __init__(self,name,reaction_string):
        self.name=name
        self.r_stoich=dict()
        self.net_stoich=dict()
        self.reaction_string=None
        s_to_react=Reaction.string_to_reaction(reaction_string)
        if s_to_react:
            self.r_stoich,self.net_stoich=s_to_react
            self.reaction_string=reaction_string
        
    def get_mol_r_stoich(self,name):
        if name in self.r_stoich:
            return self.r_stoich[name]
        return 0
    
    def get_mol_net_stoich(self,name):
        if name in self.net_stoich:
            return self.net_stoich[name]
        return 0
   
    def get_molecule_names(self):
        return set(self.r_stoich.keys()).union(set(self.net_stoich.keys()))


class Reactions:
    
    def __init__(self,use='jnp'):
        self.reactions={}
        self.molecule_i={} #molecule name to index
        self.reaction_i={} #reaction name to index
        self._use=use
        
    def _update(self):
        mol_names=set()
        
        for key,val in self.reactions.items():
            mol_names=mol_names.union(val.get_molecule_names())
        
        self.molecule_i={} #molecule name to index
        for i,k in enumerate(mol_names):
            self.molecule_i[k]=i
        
        self.reaction_i={}
        for i,key in enumerate(self.reactions.keys()):
            self.reaction_i[key]=i
    
    def add(self,name,reaction_string):
        if name in self.reactions:
            return None
        r_ok=Reaction(name,reaction_string)
        if not r_ok.reaction_string:
            return None
        self.reactions[name]=r_ok
        self._update()
        return r_ok
    
    def get_conc_as_vec(self,molecule_conc):
        if self._use=='jnp':
            return jnp.stack([0 if key not in molecule_conc else molecule_conc[key] for key in self.molecule_i.keys()])
        return np.stack([0 if key not in molecule_conc else molecule_conc[key] for key in self.molecule_i.keys()])
    
    def get_conc_as_dict(self,molecule_conc):
        conc_as_dict=dict()
        for k,i in self.molecule_i.items():
            if i<len(molecule_conc):
                conc_as_dict[k]=molecule_conc[i]
            else:
                conc_as_dict[k]=0.0
        return conc_as_dict
    
    def get_rates_as_vec(self,rates):
        if self._use=='jnp':
            return jnp.stack([0.0 if key not in rates else rates[key] for key in self.reaction_i.keys()])
        return np.stack([0.0 if key not in rates else rates[key] for key in self.reaction_i.keys()])
    
    def get_rates_as_dict(self,rates):
        rates_as_dict=dict()
        for k,i in self.reaction_i.items():
            if i<len(rates):
                rates_as_dict[k]=rates[i]
            else:
                rates_as_dict[k]=0.0
        return rates_as_dict
        
    def __getitem__(self, name):  # overload [] operator
        return self.reactions[name]

    def __iter__(self):
        return iter(self.reactions)

    def __len__(self):
        return len(self.reactions)
    
    def print_stochiometry(self):
        for name,react in self.reactions.items():
            print(name+':')
            print('--------')
            for mol,p in react.r_stoich.items():
                print('Reactant {0}  has power {1}'.format(mol,p))
            for net_mol,net_mult in react.net_stoich.items():
                print('{0} appears with multiplicity {1}'.format(net_mol,net_mult))
            print('========')


class ODE:
    
    def __init__(self,reactions):
        self.R=reactions
        self._use=reactions._use
        self.concentration_functions=dict()
        self.reaction_functions=dict()
        self._remove_rate_index=dict()
        self.dyn_molecule_i=dict() #map from dynamic molecules (those governed by ODE) to index 
        self.dyn_reaction_i=dict() #map from dynamic reactions (those governed by ODE) to index
        self._update() 
        
        
    def get_dyn_conc_as_vec(self,molecule_conc):
        if self._use=='jnp':
            return jnp.stack([0 if key not in molecule_conc else molecule_conc[key] for key in self.dyn_molecule_i.keys()])
        return np.stack([0 if key not in molecule_conc else molecule_conc[key] for key in self.dyn_molecule_i.keys()])
        
    def get_dyn_conc_as_dict(self,molecule_conc):
        conc_as_dict=dict()
        for k,i in self.dyn_molecule_i.items():
            if i<len(molecule_conc):
                conc_as_dict[k]=molecule_conc[i]
            else:
                conc_as_dict[k]=0.0
        return conc_as_dict
    
    def get_dyn_rates_as_vec(self,rates):
        if self._use=='jnp':
            return jnp.stack([0.0 if key not in rates else rates[key] for key in self.dyn_reaction_i.keys()])
        return np.stack([0.0 if key not in rates else rates[key] for key in self.dyn_reaction_i.keys()])
    
    def get_dyn_rates_as_dict(self,rates):
        rates_as_dict=dict()
        for k,i in self.dyn_reaction_i.items():
            if i<len(rates):
                rates_as_dict[k]=rates[i]
            else:
                rates_as_dict[k]=0.0
        return rates_as_dict
    
    
    def _update(self):
        self.dyn_molecule_i=dict()
        for k in self.R.molecule_i.keys():
            if k not in self.concentration_functions:
                self.dyn_molecule_i[k]=len(self.dyn_molecule_i)
        
        self.dyn_reaction_i=dict()
        for k in self.R.reaction_i.keys():
            if k not in self.reaction_functions or not self._remove_rate_index[k]:
                self.dyn_reaction_i[k]=len(self.dyn_reaction_i)
        
        if self._use=='jnp':
            embed_dyn=jnp.zeros((len(self.R.molecule_i),len(self.R.molecule_i)-len(self.concentration_functions)))
            for k,i in self.dyn_molecule_i.items():
                embed_dyn=embed_dyn.at[self.R.molecule_i[k],i].set(1.0)
        else:
            embed_dyn=np.zeros((len(self.R.molecule_i),len(self.R.molecule_i)-len(self.concentration_functions)))
            for k,i in self.dyn_molecule_i.items():
                embed_dyn[self.R.molecule_i[k],i]=1.0

        self._embed_dyn_val=embed_dyn
  
        if self._use=='jnp':
            embed_func=jnp.zeros((len(self.R.molecule_i),len(self.concentration_functions)))
            for i,k in enumerate(self.concentration_functions.keys()):
                embed_func=embed_func.at[self.R.molecule_i[k],i].set(1.0)
        else:
            embed_func=np.zeros((len(self.R.molecule_i),len(self.concentration_functions)))
            for i,k in enumerate(self.concentration_functions.keys()):
                embed_func[self.R.molecule_i[k],i]=1.0

        self._embed_func_val=embed_func
        
    def add_concentration_function(self,molecule,function): #function that maps from time to concentration value for a given molecule
        if not molecule in self.R.molecule_i:
            return
        self.concentration_functions[molecule]=function
        self._update()
    
    def add_reaction_function(self,reaction,function,remove_rate_index=True): #function that maps from time to reaction rate for a given reaction
        if not reaction in self.R.reactions:
            return
        self.reaction_functions[reaction]=function
        self._remove_rate_index[reaction]=remove_rate_index
        self._update()
    
    def dy_dt(self,y,t,*rates):
        if self._use=='jnp':
            return self.dy_dt_jnp(y,t,*rates)
        return self.dy_dt_np(y,t,*rates)
    
    def dy_dt_jnp(self,y,t,*rates):
        #print("0")
        conc=jnp.dot(self._embed_dyn_val,y)
        if len(self.concentration_functions)>0:
            f_conc=jnp.stack([fun(t) for fun in self.concentration_functions.values()]) 
            #can be expanded to be function of concentrations as well
            conc=conc+jnp.dot(self._embed_func_val,f_conc)
        
        #print(conc)
        #print("1")
        i=0
        dydt_val=jnp.zeros(len(self.dyn_molecule_i))
        s_rates, =rates  #rates[...,:]
        #print("2")
        #print(s_rates)
        for key,react in self.R.reactions.items(): #loop over reactions
            if key in self.reaction_functions:
                thisrate=self.reaction_functions[key](t)
                if not self._remove_rate_index[key]:
                    i=i+1
                #can be expanded to be function of concentrations as well
            else:
                thisrate=s_rates[i] #s_rates[i]
                i=i+1
                #print("Doing reaction {}".format(i))
                for mol,p in react.r_stoich.items():
                    thisrate=thisrate*conc[self.R.molecule_i[mol]]**p #law of mass action
            for net_mol,net_mult in react.net_stoich.items():
                if net_mol in self.dyn_molecule_i:
                    dyn_i=self.dyn_molecule_i[net_mol]
                    dydt_val=dydt_val.at[dyn_i].add(net_mult*thisrate)
        #print("Returning dydt_val...")
        return dydt_val
    
    def dy_dt_np(self,y,t,*rates):
        conc=np.dot(self._embed_dyn_val,y)
        if len(self.concentration_functions)>0:
            f_conc=np.stack([fun(t) for fun in self.concentration_functions.values()]) 
            #can be expanded to be function of concentrations as well
            conc=conc+np.dot(self._embed_func_val,f_conc)
            
        i=0
        dydt_val=np.zeros(len(self.dyn_molecule_i))
        s_rates=rates  #rates[...,:]

        for key,react in self.R.reactions.items(): #loop over reactions
            if key in self.reaction_functions:
                thisrate=self.reaction_functions[key](t)
                if not self._remove_rate_index[key]:
                    i=i+1
                #can be expanded to be function of concentrations as well
            else:
                thisrate=s_rates[i] #s_rates[i]
                i=i+1
                for mol,p in react.r_stoich.items():
                    thisrate=thisrate*conc[self.R.molecule_i[mol]]**p #law of mass action
            for net_mol,net_mult in react.net_stoich.items():
                if net_mol in self.dyn_molecule_i:
                    dyn_i=self.dyn_molecule_i[net_mol]
                    dydt_val[dyn_i]+=net_mult*thisrate
        #print("Returning dydt_val...")
        return dydt_val


# Helper functions to set concentration profiles

def get_heaviside_fun(threshold,sharpness,value=1.0,use='jnp'):
    
    def lambda_heavy_jnp(t):
        return 0.5*value*(jnp.tanh((t-threshold)/sharpness)+1)
    def lambda_heavy_np(t):
        return 0.5*value*(np.tanh((t-threshold)/sharpness)+1)
    
    if use=='jnp':
        return lambda_heavy_jnp
    return lambda_heavy_np

def get_located_fun(thresholds,sharpness,value=1.0,use='jnp'):
    def lambda_located_jnp(t):
        return 0.25*value*(jnp.tanh((t-thresholds[0])/sharpness)+1)*(jnp.tanh((thresholds[1]-t)/sharpness)+1)
    
    def lambda_located_np(t):
        return 0.25*value*(np.tanh((t-thresholds[0])/sharpness)+1)*(jnp.tanh((thresholds[1]-t)/sharpness)+1)
   
    if use=='jnp':
        return lambda_located_jnp
    return lambda_located_np

# returns a Heaviside function with threshold=threshold and sharpness=sharpness. Use negative sharpness to revert heaviside.
