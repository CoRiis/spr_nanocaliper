import numpy as np
from scipy.integrate import odeint as odeint_np
import scipy.optimize

class SPRFitter:
    def __init__(self,all_data,mdls,print_every=10):
        self.all_data=all_data
        self.mdls=mdls

        self.print_every=print_every
        self.n=np.sum([len(x.curve.t) for x in all_data])
        self.fixed_rates={}
        self.locked_rates={}
        self.bound_rates={}
        self._rate_indices_for_optim={}
        self.fixed_B=None
        self.bound_B=None
        self._optim_counter=0
        self._optim_best_B=None
        self._optim_best_rates=None
        self._optim_best_chi2=np.inf
        self.reaction_names=mdls[0].R.reactions.keys()
        self.predict_molecules={}
        self._update_rate_indices_for_optim()
        
    def add_molecule_to_predict(self,name,stoch=1):
        if name in self.mdls[0].dyn_molecule_i:
            index=self.mdls[0].dyn_molecule_i[name]
            self.predict_molecules[name]={'index': index,'stoch': stoch}
    
    @staticmethod
    def _transform_x_to_output(val,bounds=None):
        if bounds is None:
            return np.exp(val)
        lower,upper=bounds
        D=upper-lower     
        x_mean=(np.log(upper)+np.log(lower))/2
        logD=np.log(upper)-np.log(lower)     
        return lower+D*(np.tanh((val-x_mean)/logD)+1)
                                
    @staticmethod
    def _transform_output_to_x(val,bounds=None):
        if bounds is None:
            return np.log(val)
        lower,upper=bounds
        D=upper-lower     
        x_mean=(np.log(upper)+np.log(lower))/2
        logD=np.log(upper)-np.log(lower)
        return np.arctanh((val-lower)/D-1)*logD+x_mean
    
    def fix_rate(self,name,value):
        if name in self.reaction_names:
            self.fixed_rates[name]=value
        self._update_rate_indices_for_optim()
    
    def lock_rate_to(self,name,other_rate,factor=1):
        #print("calling lock_rate_to")
        if name in self.reaction_names and other_rate in self.reaction_names:
            #print('lock rate {0} to {1}'.format(name,other_rate))
            self.locked_rates[name]={'rate': other_rate,'factor': factor}
        #print('updating')
        self._update_rate_indices_for_optim()
    
    def bound_rate(self,name,bound):
        self.bound_rates[name]=bound
        
        
    def fix_B(self,value):
        self.fixed_B=value
    
    def bound_B(self,bound):
        self.bound_B=bound
    
    def _update_rate_indices_for_optim(self):
        i=0
        self._rate_indices_for_optim={}
        for name in self.reaction_names:
            if name in self.fixed_rates or name in self.locked_rates:
                continue
            #print('setting _rate_index for {0} to {1}'.format(name,i))
            self._rate_indices_for_optim[name]=i
            i=i+1
    
    def fix_B(self,value):
        self.fixed_B=value
    
    def get_free_rates_for_optimization(self,guess_rates): #replaces get_actual_rates
        actual_rates={}
        for r in self.reaction_names:
            if r not in self.fixed_rates and r not in self.locked_rates:
                actual_rates[r]=guess_rates[r]
        return actual_rates
    
    def _get_all_rates_from_input(self,x):
        lrates=[]
        for r in self.reaction_names:
            if r in self.fixed_rates:
                lrates.append(self.fixed_rates[r])
            else:
                if r in self.locked_rates:
                    rate_i=self._rate_indices_for_optim[self.locked_rates[r]['rate']]
                    fac=self.locked_rates[r]['factor']
                else:
                    rate_i=self._rate_indices_for_optim[r]
                    fac=1.0
                bounds=None if r not in self.bound_rates else self.bound_rates[r]
                lrates.append(SPRFitter._transform_x_to_output(x[rate_i],bounds)*fac)
        return np.array(lrates)
    
    def _get_B_and_all_rates_from_input(self,x):
        if self.fixed_B is not None:
            B=self.fixed_B
            starti=0
        else:
            B=SPRFitter._transform_x_to_output(x[0],bounds=self.bound_B)
            starti=1
        rates=self._get_all_rates_from_input(x[starti:])
        return (B,rates)
    
     
    def get_x0(self,max_signal,guess_rates_dict):
        if self.fixed_B is None:
            bounds=self.bound_B
            x0=[SPRFitter._transform_output_to_x(max_signal,bounds)]
        else:
            x0=[]
        actual_rates=self.get_free_rates_for_optimization(guess_rates_dict)
        for k,v in actual_rates.items():
            bounds=None if k not in self.bound_rates else self.bound_rates[k]
            x0.append(SPRFitter._transform_output_to_x(v,bounds))
        x0=np.array(x0)   
        return x0
    
    def predict(self,z):
        ret=0
        for key,val in self.predict_molecules.items():
            ret+=val['stoch']*z[:,val['index']]
        return ret
        
    def SquareLoss(self,x):
        #print(x)
        B,rates=self._get_B_and_all_rates_from_input(x)
        #print('B={0}, rate={1}'.format(B,rates))
        y0=self.mdls[0].get_dyn_conc_as_vec({'B': B})
        youts=[]
        ydata=[]
        
        chi2=0
        for i in range(len(self.mdls)):
            t=self.all_data[i].curve.t
            z=odeint_np(self.mdls[i].dy_dt, y0, t, (*rates,))
            z_predict=self.predict(z)
            chi2+=np.sum((z_predict-self.all_data[i].curve.y)**2)
        chi2=chi2/self.n
        
        #store best values
        if chi2<self._optim_best_chi2:
            self._optim_best_chi2=chi2
            self._optim_best_B=B
            self._optim_best_rates=rates
        
        if not self.print_every is None and self._optim_counter % self.print_every ==0 :
            SPRFitter._print_current(self._optim_counter,B,self.mdls[0].get_dyn_rates_as_dict(rates),chi2)
        self._optim_counter+=1
        
        return chi2
    
    @staticmethod
    def _print_current(i,B,rates,chi2):
        rates_str=""
        for k,v in rates.items():
            rates_str=rates_str+k+'='+str(v)+', '
        print("i={0}: B = {1}, rates=({2}), chi2={3}".format(i,B,rates_str,chi2))
        
    
    def minimize(self,max_signal,guess_rates_dict,method='Nelder-Mead',**kwargs):
        #print(max_signal)
        #print(guess_rates_dict)
        x0=self.get_x0(max_signal,guess_rates_dict)
        #print(x0)
        self._optim_best_chi2=np.inf
        self._optim_counter=0
        res=scipy.optimize.minimize(self.SquareLoss,x0=x0,method=method,**kwargs)
        best_B=self._optim_best_B
        best_rates=self._optim_best_rates
        best_rates_dict=self.mdls[0].get_dyn_rates_as_dict(best_rates)
        best_chi2=self._optim_best_chi2
        if self.print_every is not None:
            SPRFitter._print_current(self._optim_counter,best_B,best_rates_dict,best_chi2)
        
        #B,rates=self._get_B_and_all_rates_from_input(res.get('x'))
        return {'B': best_B,'rates': best_rates,'rates_dict': best_rates_dict,'chi2': best_chi2,'iterations': self._optim_counter,'res': res} #,'res_B': B,'res_rates': rates}
    
    def __Curves_fitting(self,tdata,*param):
        B=np.exp(param[0])
        y0=self.mdls[0].get_dyn_conc_as_vec({'B': B})
        rates=np.exp(param[1:])
        curve = None
        for i in range(len(self.mdls)):
            t=self.all_data[i].curve.t
            z=odeint_np(self.mdls[i].dy_dt, y0, t, (*rates,))
            z_observed=z[:,self.AB_i].reshape((len(t)))
            if curve is None: 
                curve = z_observed
            else: 
                curve = np.concatenate((curve,z_observed))
        return curve
