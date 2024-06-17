import pandas as pd
import numpy as np
import re

class Curve:
    def __init__(self):
        t=None
        y=None
        
    def remove_nan(self):
        ri,=np.where(np.isnan(self.y))
        self.t=np.delete(self.t,ri)
        self.y=np.delete(self.y,ri)


class SPR_Data:
    def __init__(self,conc=0.0):
        self.metadata = {"concentration":conc}
        self.curve =Curve()

def read_SPR_data(file,remove_nan=True):
    data = pd.read_excel(file,engine='openpyxl')
    data = data.astype(str)
    data = data.stack().str.replace(',','.').unstack()
    data = data.astype(float)
    all_data=[]
    columns=data.columns
    for i,c in enumerate(columns):
        m=re.search('[0-9.]+ nM_X$',c)
        if m is not None: 
            conc=float(m.group(0)[:-4])
            all_data.append(SPR_Data(conc))
            all_data[-1].curve.t=np.array(data[columns[i]])
            all_data[-1].curve.y=np.array(data[columns[i+1]])
            if remove_nan:
                all_data[-1].curve.remove_nan()
    return all_data
