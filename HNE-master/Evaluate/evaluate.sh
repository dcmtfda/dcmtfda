#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' support attributed='True'

dataset='PubMed' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
#model='R-GCN' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', and 'ConvE'
model='HIN2Vec'
#model='ConvE'
task='both' # choose 'nc' for node classification, 'lp' for link prediction, or 'both' for both tasks
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

##

#ConvE HIN2Vec R-GCN 
#metapath2vec-ESim HIN2Vec
for model in HIN2Vec metapath2vec-ESim
do
	python evaluate.py -dataset ${dataset} -model ${model} -task ${task} -attributed ${attributed} -supervised ${supervised}
done



