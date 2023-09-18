import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

def synthetic_data_two_var(seed,R=0.5):
    np.random.seed(seed)
    L = 1000
    X_INIT = 10
    Y_INIT = 10
    # R = 0.5
    X_NOISE = np.random.normal(0,1,L)
    Y_NOISE = np.random.normal(0,1,L)
    X = np.zeros(L)
    Y = np.zeros(L)
    X[0] = X_INIT
    Y[0] = Y_INIT
    for i in range(1,L):
        X[i] = X[i-1] + X_NOISE[i]
        Y[i] = (1-R)*Y[i-1] + R*X[i-1] + Y_NOISE[i]

    # fig, ax = plt.subplots(figsize = (10,5))
    # ax.plot(np.arange(L),X,label='X')
    # ax.plot(np.arange(L),Y,label='Y')
    # ax.legend()
    # # plt.show()
    # plt.savefig('./result/syn_XY_R05_seed0.pdf')
    X = X[1:]-X[:-1]
    Y = Y[1:]-Y[:-1]

    return X,Y

def heatmap_2d(matrix,labels,savepath):
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix,cmap = 'jet')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_title("2D Heatmap")
    fig.tight_layout()
    fig.colorbar(im,ax=ax)
    # plt.show()
    plt.savefig(savepath)
    pass

def histogram(x1,x2,x3,BINS,savepath):
    fig = plt.figure(tight_layout=True,figsize=(10,6))
    gs = gridspec.GridSpec(2, 2)

    counts1, bins1 = np.histogram(x1,bins=BINS)
    counts2, bins2 = np.histogram(x2,bins=BINS)
    counts3, bins3 = np.histogram(x3,bins=BINS)
    
    ax = fig.add_subplot(gs[1, :])
    ax.stairs(counts3, bins3)
    ax.set_title(r'$T_{x\rightarrow y}:T_{y\rightarrow x}$')

    ax = fig.add_subplot(gs[0, 0])
    ax.stairs(counts1, bins1)
    ax.set_title(r'$T_{x\rightarrow y}$')

    ax = fig.add_subplot(gs[0, 1])
    ax.stairs(counts2, bins2)
    ax.set_title(r'$T_{y\rightarrow x}$')
    
    fig.tight_layout(pad=1.0)
    # plt.show()
    plt.savefig(savepath)
    pass

def plot_graph(matrix,labels,sizes,savepath):
    sizes = np.array(sizes)
    labels = np.array(labels)
    sizes = 500*sizes/sizes.mean()
    matrix = matrix*matrix
    matrix = matrix/matrix.max()

    G = nx.Graph()
    G.add_nodes_from([(l,{"labels":s}) for l,s in zip(labels,sizes)])

    fig, ax = plt.subplots(figsize=(5,5))
    nx.draw_circular(G,node_size = sizes, with_labels=True)
    node_pos=nx.circular_layout(G)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i!=j:
                nx.draw_networkx_edges(G,pos = node_pos,edgelist = [(labels[i],labels[j])], alpha = matrix[i][j])
    plt.savefig(savepath)
    pass

def plot_xyR(x,TE_xyR_list,TE_yxR_list,TE_ratioR_list,savepath):
    fig = plt.figure(tight_layout=True,figsize=(10,6))
    gs = gridspec.GridSpec(2, 2)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(x,TE_ratioR_list)
    ax.set_title(r'$T_{x\rightarrow y}:T_{y\rightarrow x}$')
    ax.set_xlabel('r')

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x,TE_xyR_list)
    ax.set_title(r'$T_{x\rightarrow y}$')
    ax.set_xlabel('r')
    ax.set_ylim(1.35,1.6)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x,TE_yxR_list)
    ax.set_title(r'$T_{y\rightarrow x}$')
    ax.set_xlabel('r')
    ax.set_ylim(1.35,1.6)

    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
    # plt.show()
    plt.savefig(savepath)
    pass

if __name__=='__main__':
    plot_graph()
    # synthetic_data_two_var()