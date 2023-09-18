import numpy as np
from scipy import stats
from scipy import ndimage
import os
import pandas as pd
from tqdm import tqdm
from utils import heatmap_2d, synthetic_data_two_var, histogram, plot_graph, plot_xyR
from transfer_entropy import transfer_entropy,effective_transfer_entropy


def read_nasdaq_market_value(topk):
    filepath = './data/nasdaq_screener_1681925375421.csv'
    data_csv = pd.read_csv(filepath)
    valid_comp = os.listdir('./data/StockMarket/stock_market_data/forbes2000/csv/')
    valid_comp = [i.split('.')[0] for i in valid_comp]
    data_csv = data_csv[data_csv['Symbol'].isin(valid_comp)]
    data_csv.sort_values(by=['Market Cap'],ascending=False,inplace=True)
    data_topk = data_csv.head(topk)
    data_topk.reset_index(inplace=True)
    return data_topk

def read_nasdaq_by_sector(delay):
    # Read metadata
    filepath = './data/nasdaq_screener_1681925375421.csv'
    data_csv = pd.read_csv(filepath)
    valid_comp = os.listdir('./data/StockMarket/stock_market_data/collections/csv/')
    data_sector = {}
    minl = 2001
    i=0
    # Reading and Aggregating data
    print("Loading and aggregating company data group by sector...")
    for comp in tqdm(valid_comp):
        comp_data = pd.read_csv('./data/StockMarket/stock_market_data/collections/csv/{}'.format(comp))
        syb = comp.split('.')[0]
        if syb in list(data_csv['Symbol']) and sum(comp_data['Close']>0)>=minl:
            sector = data_csv[data_csv['Symbol']==syb]['Sector'].iloc[0]
            if type(sector)!=type(''):
                continue
            if sector not in data_sector.keys():
                data_sector[sector] = np.array(comp_data['Close'][-minl:])
            else:
                data_sector[sector] = data_sector[sector] + np.array(comp_data['Close'][-minl:])
            # if np.nan in data_sector[sector]:
            #     np.nan
            # if sector=='Real Estate':
            #     if sum(data_sector[sector])==np.nan:
            #         np.nan
    data_sector = pd.DataFrame(data_sector)
    # data_sector.to_csv('./result/cache/data_sector.csv')
    log_return = data_sector[1:].reset_index()-data_sector[:-1].reset_index()
    log_return = log_return.drop(columns=['index'])
    for k in log_return.keys():
        log_return[k] = (log_return[k]-log_return[k].mean())/(log_return[k].std())
    # Transfer entropy
    transfer = np.zeros((len(log_return.keys()),len(log_return.keys())))
    print("Calculating transfer matrix")
    for i,ki in tqdm(enumerate(log_return.keys())):
        for j,kj in tqdm(enumerate(log_return.keys()),leave=False):
            transfer[i][j] = transfer_entropy(log_return[ki],log_return[kj],delay=delay)
    market_cap = np.array(np.log(data_sector.mean()))
    return transfer,log_return.keys(),market_cap

def transfer_matrix(data,delay):
    data_mv = pd.DataFrame({})
    minl = 2484
    for i in range(len(data)):
        comp_syb = data['Symbol'][i]
        comp_data = pd.read_csv('./data/StockMarket/stock_market_data/forbes2000/csv/{}.csv'.format(comp_syb))
        # data_mv[comp_syb] = comp_data['Close'][-minl:].reset_index()['Close']
        data_mv[comp_syb] = np.array(np.log(comp_data['Close'][-minl:])[1:]) \
                        -np.array(np.log(comp_data['Close'][-minl:])[:-1])
    data_date = comp_data['Date'][-1:]
    transfer = np.zeros((len(data),len(data)))
    syb = data['Symbol']
    print("Calculating transfer matrix")
    for i in tqdm(range(len(data))):
        for j in tqdm(range(len(data)),leave=False):
            transfer[i][j] = transfer_entropy(data_mv[syb[i]],data_mv[syb[j]],delay=delay)
    return transfer

if __name__=='__main__':
    # # NASDAQ top company
    # data_topk = read_nasdaq_market_value(topk=20)
    # transfer = transfer_matrix(data_topk,delay=1)
    # np.fill_diagonal(transfer, 0)
    # # np.save("./result/cache/transfer_top20.npy",transfer)
    # # data_topk.to_csv("./result/cache/data_topk.csv")
    # transfer = np.load("result/cache/transfer_top20.npy",allow_pickle=True)
    # data_topk = pd.read_csv("./result/cache/data_topk.csv")
    # heatmap_2d(transfer,data_topk['Symbol'],savepath = './result/heatmap_2d.png')
    # plot_graph(transfer,data_topk['Symbol'],data_topk['Market Cap'],savepath = './result/graph_top20.png')
    # # data_topk.to_csv('./result/data_topk.csv')
    
    # # NASDAQ by sector
    # transfer,labels,market_cap = read_nasdaq_by_sector(delay=1)
    # np.fill_diagonal(transfer, 0)
    # np.save("./result/cache/sector_transfer.npy",transfer)
    # np.save("./result/cache/sector_labels.npy",labels)
    # np.save("./result/cache/sector_market_cap.npy",market_cap)
    # transfer = np.load("./result/cache/sector_transfer.npy",allow_pickle=True)
    # labels = np.load("./result/cache/sector_labels.npy",allow_pickle=True)
    # market_cap = np.load("./result/cache/sector_market_cap.npy",allow_pickle=True)
    # transfer,labels,market_cap = np.load("sector.npy")
    # heatmap_2d(transfer,labels,savepath = './result/heatmap_groupby_sector_2d.png')
    # plot_graph(transfer,list(labels),market_cap,savepath = './result/graph_sector.png')

    # Synthetic data test
    TE_xyR_list,TE_yxR_list, TE_ratioR_list = [],[],[]
    R_list = [i/10 for i in range(1,11)]
    for R in tqdm(R_list):
        TE_xy_sum=TE_yx_sum=TE_ratio_sum=0
        for seed in tqdm(range(100),leave=False):
            X,Y = synthetic_data_two_var(seed=seed,R=R)
            TE_xy = transfer_entropy(X,Y)
            TE_yx = transfer_entropy(Y,X)
            TE_xy_sum += TE_xy
            TE_yx_sum += TE_yx
            TE_ratio_sum += TE_xy/TE_yx
        TE_xy_mean = TE_xy_sum/100
        TE_yx_mean = TE_yx_sum/100
        TE_ratio_mean = TE_ratio_sum/100
        TE_xyR_list.append(TE_xy_mean)
        TE_yxR_list.append(TE_yx_mean)
        TE_ratioR_list.append(TE_ratio_mean)
    plot_xyR(np.array(R_list),TE_xyR_list,TE_yxR_list,TE_ratioR_list,savepath = './result/syn_result.png')


    # TE_xy_list,TE_yx_list,TE_ratio_list=[],[],[]
    # for seed in tqdm(range(0,1000)):
    #     X,Y = synthetic_data_two_var(seed)
    #     TE_xy = transfer_entropy(X,Y)
    #     TE_yx = transfer_entropy(Y,X)
    #     TE_ratio = TE_xy/TE_yx
    #     TE_xy_list.append(TE_xy)
    #     TE_yx_list.append(TE_yx)
    #     TE_ratio_list.append(TE_ratio)
    # TE_ratio_list = np.array(TE_ratio_list)
    # ratio = (TE_ratio_list>1).sum()/len(TE_ratio_list)
    # print(ratio)
    # histogram(TE_xy_list,TE_yx_list,TE_ratio_list,50,savepath = './result/syn_single.png')
    

    # ETE_xy = effective_transfer_entropy(X,Y,repeat=10)
    # ETE_yx = effective_transfer_entropy(Y,X,repeat=10)

    pass
