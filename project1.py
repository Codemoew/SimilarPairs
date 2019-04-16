#%%
#pro1
#read original data from Netflix_data.txt

with open('Netflix_data.txt', 'r') as f:
    data = f.readlines()
f.close()

#build a dictionary with userID as keys and the movies they rated and the ratings as values
movie_ID=0
dict={}
for line in data:
    if ':' in line:
        movie_ID+=1
    else:
        user_ID = line.split(',')[0]
        rating=line.split(',')[1]
        #only keep movies that are rated greater than or equal to 3
        if (rating=='3')|(rating=='4')|(rating=='5'):
            dict.setdefault(user_ID,[]).append([movie_ID,rating])
#delele variable 
del data

#delete the key-value pair if the key corresponds to more than 20 values
key_list=[]
for key, value in dict.items():
    if len(value)>20:
        key_list.append(key)
for x in key_list:
    del dict[x]

#create the matrix
import numpy as np
#number of movies
M=movie_ID
#number of users
N=len(dict.keys())
user=0
map=[]
matrix_result = np.zeros(shape=(M,N))
for key,value in dict.items():
    for i in range(len(value)):
        matrix_result[value[i][0]-1][user]=1
    map.append([key,user])
    user+=1
    
del dict

#%%
#Pro2
#define a function to compute the jaccard distance between two sets
def jaccard_distance(set1,set2):
    set_and=[]
    set_or=[]
    for (e1,e2) in list(zip(set1,set2)):
        set_and.append(e1 and e2)
        set_or.append(e1 or e2)
    d=1-set_and.count(1)/set_or.count(1)
    return d

dist=[]
for r in range(10000):
    #generate 10000 random pairs
    randindex=np.random.choice(N, 2, replace=False)
    randcol_1=matrix_result[:,randindex[0]]
    randcol_2=matrix_result[:,randindex[1]]
    #compute jaccard distance between the pair
    dist.append(jaccard_distance(randcol_1,randcol_2))
dist_min=min(dist)
 
#Plot Pairwise Jaccard Distance Histogram
import matplotlib.pyplot as plt
plt.hist(dist, bins=100)
plt.xlabel('Jaccard Distance')
plt.ylabel('Frequency')
plt.title('Pairwise Jaccard Distance Histogram')
#%%
#Pro3
one_index=[]
for i in range(matrix_result.shape[1]):
    one_index.append(list(np.where(matrix_result[:,i]==1)[0]))
#%%
#Pro4
#transform the original matrix into a signature matrix using hash functions
import random
#choose a large prime number 4507
#generate 1000 hash functions
hf=161
rand_a=np.array(random.sample(range(1,4507), hf))
rand_b=np.array(random.sample(range(1,4507), hf))
#build a permutation table
p=np.zeros([hf,M])
for r in range(hf):
    p[r,:]=np.arange(M)
table=(rand_a*p.T+rand_b)%4507
#build the signature matrix
sig=np.ones([hf,N])*np.inf
for i in range(N):
    for j in range(len(one_index[i])):
        sig[:,i]=np.minimum(table.T[:,one_index[i][j]],sig[:,i])

#hash the band
band=23
r=7
rand_A=np.array(random.sample(range(1,4507), hf))
rand_B=np.array(random.sample(range(1,4507), hf))
sig_hash_values=(rand_A*sig.T+rand_B)%4507
Bucket=[]
for b in range(band):
    hash_band=sig_hash_values.T[b*r:(b+1)*r]
    hash_table=hash_band.sum(0) 
    d = {}
    for col_index,item in enumerate(hash_table):
        d.setdefault(item,[]).append(col_index)
    for l in d.values():
        if len(l)>1:
            Bucket.append(l)

#%%
import itertools

def similarity(a,b):
    a=set(a)
    b=set(b)
    s=len(a.intersection(b))/len(a.union(b))
    return s
       
sim_pairs=[]
#filter out duplicates
Bucket.sort()
Bucket=list(b for b,_ in itertools.groupby(Bucket))

for item in Bucket:
    for sims in itertools.combinations(item,2):
        #Use efficient representation from Pro3
        if similarity(one_index[sims[0]],one_index[sims[1]])>0.65:
            sim_pairs.append([sims[0],sims[1]])
sim_pairs.sort()
final_pairs=list(s for s,_ in itertools.groupby(sim_pairs))
#%%
#
import csv
with open('similarPairs.csv','w') as writeFile:
    similarWriter = csv.writer(writeFile, delimiter=',')
    
    for i in range(len(final_pairs)):
        similarWriter.writerow([final_pairs[i][0], final_pairs[i][1]])

#%%
#Pro5
#find the nearest neighbor(s) of an input list of movie IDs
def NNs(seq):
    cdds={}
    for c,item in enumerate(one_index):
        s=similarity(seq,item)
        if s>0.65:
            cdds.setdefault(s,[]).append(c)
    cdd=[]
    for value in cdds.values():
        cdd=np.concatenate((cdd,value))   
    if len(cdd)==0:
        print('None found') 
        return('nan','nan')
    else:
        nns=cdds[max(cdds.keys())]    
        #returns the nearest neighbor and all the similar pairs
        return (nns,cdd)

z=NNs([5, 312, 570])
