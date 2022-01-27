## Discordance Analysis in Heterogeneous Data Collections

Source code and data used in the experiments.

## 1) Multi-way Clustering through Deep Collective Matrix Tri-Factorization (DCMTF)

Script to perform clustering before Discordance Analysis. Pick <dataset_id> based on the Case Study of interest.
In main_*.py, set `is_gpu` to `False` to run using CPU and change `gpu_id` as required when using GPU.

#### DCMTF

    `$ python -u main_dcmtf_clust.py <dataset_id> &> out.log`

##### Parameters:

|<dataset_id>|Description|
| ------ | ------ |
| "ade1" | Adverse Drug Event Identification dataset, 3 matrices, sample 1 - results in Table 1 of "DA for Knowledge Discovery"|
| "ade2" | Adverse Drug Event Identification dataset, 3 matrices, sample 2 - results in Appendix B.2|
| "ade3" | Adverse Drug Event Identification dataset, 3 matrices, sample 3 - results in Appendix B.2|
| "ade4" | Adverse Drug Event Identification dataset, 3 matrices, sample 4 - results in Appendix B.2|
| "pubmed_heuristic" | HIN dataset, 10 matrices, results in Table 3 of "DA for Data Cleaning"|
| "wiki1" | Synchronizing Wikipedia Infoboxes dataset, 3 matrices, sample 1 - results in Appendix B.1|
| "wiki2" | Synchronizing Wikipedia Infoboxes dataset, 3 matrices, sample 2 - results in Appendix B.1|
| "wiki3" | Synchronizing Wikipedia Infoboxes dataset, 3 matrices, sample 3 - results in Appendix B.1|
| "wiki4" | Synchronizing Wikipedia Infoboxes dataset, 3 matrices, sample 4 - results in Appendix B.1|

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations, Entity cluster indicators, Cluster associations at `./out_clust/<dataset_id>/`


## 1) DA Case Study: Adverse Drug Event Identification

Steps to reproduce results in Table 1 of "DA for Knowledge Discovery" and Appendix B.2

`Step 1:`  Obtain U, I and A using DCMTF for dataset "ade1"

`Step 2:`  Perform DA using the corresponding ipy notebook. Open using Jupyter and run all cells. 

#### DCMTF + DA

1. `$ python -u main_dcmtf_clust.py "ade1" &> out.log`
    
2. `da/ade/"1 - DA ade - DCMTF.ipynb"`

##### *Note*:
- To repeat this experiment for other ade datasets *ade2/ade3/ade4*: Run DCMTF for the required ade[run_no] in `Step 1` and change the variable `run_no` accordingly to *2/3/4* in the ipython notebook before running them in `step 2`.
- To run only the DA, using the previously clustered output(`dcmtfda/out_clust/ade1/`), skip step 1 above and run step 2 directly.

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations (U), Entity cluster indicators (I), Cluster associations (A) at `./out_clust/ade1/`
*  Discordant Cluster Chain and the % entities 


## 2) DA Case Study: Improving Network Representation Learning

Steps to reproduce results in Table 3 of "DA for Data Cleaning". To obtain the 2 "cleaned" versions of the PubMed HIN from the `Original HIN` viz. `Rand-Cleaned` and `DA-Cleaned`:

1. Obtain U, I and A using DCMTF for dataset "pubmed_heuristic"
	
	`$ python -u main_dcmtf_clust.py "pubmed_heuristic" &> out.log`

2. Open using Jupyter and run all cells. Performs DA, obtains edge sets (i) E: from discordant chains, (ii) R: randomly selected

	`da/pubmed/"DA - HIN - step 1 - find high scoring and random cluster chains.ipynb"`

3. Open using Jupyter and run all cells. Filters E and R from `link.data` file of `Original HIN` to produce the cleaned up `link.data` files for `DA-Cleaned` and `Rand-Cleaned` in the folders `dcmtfda/out_clust/pubmed_heuristic/version_2/cc/PubMed_da_link` and `dcmtfda/out_clust/pubmed_heuristic/version_2/cc/PubMed_rand_link` respectively.
	
	`da/pubmed/"DA - HIN - step 2 - filter and obtain cleaned network.ipynb"`

4. Make folder `dcmtfda/HNE-master/Data`. Copy `PubMed_orig/*` to `dcmtfda/HNE-master/Data/PubMed`. Proceed to steps for HIN2Vec, Metapath2Vec + DA to obtain the results for the original HIN dataset. 

5. Make folder `dcmtfda/HNE-master/Data`. Copy `PubMed_orig/*` to `dcmtfda/HNE-master/Data/PubMed`. Replace `link.data` from `cc/PubMed_da_link/link.dat` or `cc/PubMed_rand_link/link.dat` depending on whether to run the HNE methods for *DA-Cleaned HIN* and *Rand-Cleaned HIN* respectively. Proceed to steps for HIN2Vec, Metapath2Vec + DA to obtain the results for the DA cleaned/rand cleaned HIN dataset. 


#### *Note:* 
- To run only the HNE methods on DA cleaned HIN, the folder `dcmtfda/data_hin` contains the filtered version of the HINs `PubMed_orig`, `PubMed_da`, `PubMed_rand` used in our experimentation i.e. the output from the previous steps. 


#### HIN2Vec, Metapath2Vec + DA

Steps to learn the HIN2Vec/Metapath2Vec embeddings for the HINs `PubMed_orig`, `PubMed_da`, `PubMed_rand` and obtain results on the two benchmark tasks (node classification and link prediction):

1. Copy the contents of `/data_hin/PubMed_orig/*` to `/HNE-master/Data/PubMed`
2. Do `cd HNE-master/Transform` and run `$sh transform.sh` 
3. Do `cd HNE-master/Model/HIN2Vec` and run `$sh run.sh` to learn the HIN2Vec embeddings
4. Do `cd HNE-master/Model/metapath2vec-ESim` and run `$sh run.sh` to learn the Metapath2Vec embeddings
5. Do `cd HNE-master/Evaluate` and run `$sh evaluate.sh` to obtain perform benchmark tasks and record results at: `/HNE-master/Data/PubMed/record.dat`
6. Repeat the above steps 1 to 5 for `PubMed_da` and `PubMed_rand`

More details about the baselines HIN2Vec, Metapath2Vec execution can be found [here](https://github.com/yangji9181/HNE)


## 3) DA Case Study: Synchronizing Wikipedia Infoboxes

Steps to reproduce results in Appendix B.1:

`Step 1:`  Obtain U, I and A using DCMTF for dataset "wiki1"

`Step 2:`  Perform DA using the corresponding ipy notebook. Open using Jupyter and run all cells. 

#### DCMTF + DA

1. `$ python -u main_dcmtf_clust.py "wiki1" &> out.log`
    
2. `da/wiki/"1 - DA wiki - DCMTF.ipynb"`

##### *Note*:
- To repeat this experiment for other wiki datasets *wiki2/wiki3/wiki4*: Run DCMTF for the required wiki[run_no] in `Step 1` and change the variable `run_no` accordingly to *2/3/4* in the ipython notebook before running them in `step 2`. 

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations (U), Entity cluster indicators (I), Cluster associations (A) at `./out_clust/wiki1/`
*  Discordant Cluster Chain and the % entities found

## Other Clustering methods' source code:

- CFRM: Collective Factorization of Related Matrices

	`$ python -u main_cfrm_clust.py <dataset_id>  &> out.log`

- DFMF: Data Fusion by Matrix Factorization 
	
	`$python -u main_dfmf_clust.py <dataset_id>  &> out.log`

- The DA `* DA * - DCMTF.ipynb` source code can be made to work with CFRM/DFMF by pointing to the corresponding output directory and changing the filenames as needed.

## Prerequisites
- DCMTF: [Python37, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)
- CFRM,DFMF: [Python27, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)
- DA: [NetworkX](https://networkx.org/)
- HIN2Vec: Python37, other details [here](https://github.com/yangji9181/HNE/tree/master/Model/HIN2Vec)
- Metapath2Vec: Python37, requires 2 external packages, details [here](https://github.com/yangji9181/HNE/tree/master/Model/metapath2vec-ESim) 


