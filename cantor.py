#!/usr/bin/env python3
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import time
from collections import Counter, defaultdict
import csv
import sys
import hnswlib

model_path = sys.argv[1]
data_path = sys.argv[2]
NUM_FINAL_LIST = int(sys.argv[3])
num_ef = int(sys.argv[4])
NUM_TOP_K = int(sys.argv[5])
output_path = sys.argv[6] + '.{}.top{}'.format(NUM_FINAL_LIST, NUM_TOP_K)

lol = list(csv.reader(open(model_path + '.p', 'r'), delimiter='\t'))

d = []
for dd in lol:
    p = list(map(float, dd[0].split(" ")))
    d.append(p)
ppl = np.asarray(d, dtype=np.float32)

lol = list(csv.reader(open(model_path + '.q', 'r'), delimiter='\t'))
d = []
for dd in lol:
    p = list(map(float, dd[0].split(" ")))
    d.append(p)
item = np.asarray(d, dtype=np.float32)

np.random.seed(252)
NUM_PERM = 128
num_P, num_Q = 0, 0
deg_P, deg_Q = Counter(), Counter()

# Load data.
raw_edges = []
with open(data_path, 'r') as fp:
    for line in fp:
        pi, qj, _ = map(int, line.split(' '))
        raw_edges.append((pi, qj))
        num_P = max(num_P, pi + 1)
        num_Q = max(num_Q, qj + 1)
        deg_P[pi] += 1
        deg_Q[qj] += 1 
    
assert(len(ppl) == num_P)
assert(len(item) == num_Q)

row_norm = np.linalg.norm(ppl, axis=1)
n_x = ppl/(row_norm[:, np.newaxis] +1e-6)

ncluster = 8
W = ppl
nclass,d = W.shape
n,_ = ppl.shape
centers = np.zeros([ncluster, d], dtype=np.float32)
pi = np.random.randint(ncluster, size=n)

def adaptive_fast_clustering(item,ncluster,window):
    ncluster =  min(ncluster, len(item))
    if ncluster == 0:
        return [], []
    a2 = np.sum(item ** 2,1).reshape(-1,1)
    sorted_idx = np.argsort(a2.reshape(-1))[::-1]
    total_n = item.shape[0]
    interval = int(total_n/ncluster)
    ada_centers = np.zeros([ncluster,10])
    for i in range(ncluster):
        aaa = np.sum(item[sorted_idx[i*interval:(i+1)*interval]],0)
        ada_centers[i] = aaa/(np.linalg.norm(aaa) + 1e-6)
    ada_centers = ada_centers[np.where(np.sum(ada_centers,1) != 0)[0]]
    loss = item.dot(ada_centers.transpose())
        
    pi = np.argmax(loss,1)
    nnn = ada_centers.shape[0]
    nnnn = nnn

    prev_loss = np.sum(loss[np.arange(pi.shape[0]),pi])
    for i in range(10):
        for i in range(nnnn):
            p = np.where(pi == i)[0]
            if p.shape[0] == 0:
                ada_centers[i] = np.zeros(10)
            else:
                aaa = np.sum(item[p], 0)
                ada_centers[i] = aaa/(np.linalg.norm(aaa) + 1e-6)
        ada_centers = ada_centers[np.where(np.sum(ada_centers,1) != 0)[0]]

        loss = item.dot(ada_centers.transpose())
        pi = np.argmax(loss,1)  
        new_loss = np.sum(loss[np.arange(pi.shape[0]),pi])
        prev_loss = new_loss
        
        outlier = np.where(loss[np.arange(pi.shape[0]),pi] < 0.99)[0]
        N_new_cluseter = int(outlier.shape[0]/window) + 1
        hiha = sorted_idx[outlier]
        new_centers = np.zeros([N_new_cluseter,10])
        for i in range(N_new_cluseter):
            aaa = np.sum(item[hiha[i*interval:(i+1)*interval]], 0)
            new_centers[i] = aaa/(np.linalg.norm(aaa) + 1e-6)
        ada_centers = np.vstack([ada_centers,new_centers])
        nnnn = ada_centers.shape[0]
    ada_centers = ada_centers[np.where(np.sum(ada_centers,1) != 0)[0]]
    return ada_centers,item[outlier]

                

# Start pre-processing.

start = time.time()
print('Start!!')

smg = hnswlib.Index(space='ip', dim=len(item[0]))
smg.init_index(max_elements=num_Q, ef_construction=num_ef, M=16, random_seed=252)
smg.add_items(item, np.arange(num_Q), num_threads=1)
smg.set_ef(num_ef)

print('Built hnsw in {} seconds.'.format(time.time() - start))

P_scores = np.array([np.log(deg_P[i]) + 1.0 for i in range(num_P)], dtype=np.float32)
P_scores /= np.sum(P_scores)
P_samples = np.random.choice(num_P, NUM_FINAL_LIST, p=P_scores, replace=False)

print('Generated final list in {} seconds.'.format(time.time() - start))

idx = np.arange(ppl.shape[0])
idx = np.random.permutation(idx)
NUM = NUM_FINAL_LIST
r_x_actual = ppl[P_samples]
r_x = n_x[P_samples]
pi = np.random.randint(ncluster, size=NUM)
for it in range(100):
        obj = 0
        for j in range(ncluster):
                aaa = np.sum(r_x[pi==j, :], 0)
                centers[j] = aaa/(np.linalg.norm(aaa) + 1e-6)
                obj += np.linalg.norm(aaa)

        pi = np.argmax(np.matmul(r_x, centers.transpose()), 1)

print('Spec clustering in {} seconds.'.format(time.time() - start))

## Cluster maps 
centers = centers.transpose()
preds = np.argmax(np.matmul(r_x, centers), 1)
d_map = [0] * ncluster
KK = NUM_TOP_K

print('Now in {} seconds.'.format(time.time() - start))

item_t = item.transpose()

PER_BATCH = 1000
d_freq = [Counter() for i in range(ncluster)]
for i in range(ncluster):
    rr = r_x[np.where(preds==i)[0]]
    KK = 15
    kap = np.vstack(adaptive_fast_clustering(rr,600,10))
    rr, _ = smg.knn_query(kap, k=KK, num_threads=1)
    
    for gg in np.reshape(rr[:,:KK],(-1,)):
        d_freq[i][gg] += 1
    d_map[i] = np.unique(np.reshape(rr[:,:KK],(-1,)))
    print('Process cluster {} in {} seconds.'.format(i, time.time() - start))

print('Now in {} seconds.'.format(time.time() - start))
init_map = np.zeros([ncluster,item.shape[0]])
for i in range(ncluster):
    init_map[i][d_map[i]] = 1

print('Now in {} seconds.'.format(time.time() - start))

reduced_d_map = {}
for k,d in enumerate(d_freq):
    nn = [pde for pde in d_map[k] if d[pde] >= 1]
    reduced_d_map[k] = np.asarray(nn)

reduced_matrices = [item[reduced_d_map[rr]] for rr in range(ncluster)]
eeeend = time.time()


print("training_time",eeeend-start)


import time
AA = 0
BB = 0
CC = 0
total_time = 0
K = 20
with open(output_path, 'w') as wp:
    for epoch in range(1):
        cc = 0
        cnt = 0
        ll = 0.
        ll5 = 0.
        ll10 = 0.
        ll20 = 0.
        for k,b in enumerate(ppl):
            batch = b.reshape(1,-1)        
            
            start = time.time()
            
            a = np.matmul(batch, centers)
            rr = np.argmax(a)
            
            w2w = reduced_matrices[rr]
            results2 = np.matmul(w2w,b)
            if len(results2) < K:
                bb = np.arange(len(results2))
            else:
                bb = np.argpartition(results2,-K)[-K:]
            
            real_top_20 = d_map[rr][bb[np.argsort(results2[bb])[::-1]]]
            end = time.time()

            total_time += (end - start)

            for kk in range(20):
                if kk > 0: wp.write(' ')
                wp.write('{}'.format(real_top_20[kk] if kk < len(real_top_20) else 0))
            wp.write('\n')

        print(total_time)
