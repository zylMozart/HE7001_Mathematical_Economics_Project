import pandas as pd
import numpy as np

def table_topk():
    data = pd.read_csv('data_topk.csv')
    for i in range(len(data)):
        print(data.iloc[i,2],data.iloc[i,3],"{:.2e}".format(data.iloc[i,7]),'\\'+data.iloc[i,4].split('.')[0],data.iloc[i,10],data.iloc[i,11],sep=' & ',end='\\\\ \n')
        print('\\hline')
    pass

def table_sector():
    data = pd.read_csv('./cache/data_sector.csv')
    data = data.mean()[1:]
    n=len(data.keys())//2
    for i in range(n):
        print(data.keys()[i],"{:.2e}".format(data[i]),
        data.keys()[i+n],"{:.2e}".format(data[i+n]),
        sep = ' & ', end ="\\\\ \n")
        print('\\hline')
    pass
if __name__=='__main__':
    table_sector()
    pass