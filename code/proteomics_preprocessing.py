'''Preparing datasets for language modelling, classification and sequence annotation
'''

import fire
# This needs to be called in order to load local implemented submodules
import os
import sys
#module_path = os.path.abspath(os.path.join('../'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

from tqdm import tqdm as tqdm
from utils.proteomics_utils import *
from utils.dataset_utils import *

#for log header
import datetime
import subprocess

import itertools
import ast
from pandas import concat

#for kinase dicts
from collections import defaultdict
######################################################################################################
#DATA PATHS
######################################################################################################
#path to the data directory
data_path = Path('../data')

#SEQUENCE DATA
#ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz (this is swissprot not the whole uniprot)
path_sprot=data_path/'uniprot_sprot_2017_03.xml'

#CLUSTER DATA (default path)
#ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz unzipped 100GB file (whole uniprot not just swissprot)
path_uniref = data_path/"uniref50_2017_03.xml"
#path to the data directory also uploaded to git
git_data_path = Path('../git_data')
#path to the temporary directory for dataframes
tmp_data_path = Path("../tmp_data")#should be moved to ./data_tmp or similar
tmp_data_path.mkdir(exist_ok=True)

######################################################################################################
#AUX FUNCTIONS
######################################################################################################
def load_uniprot(source=path_sprot,parse_features=[]):
    '''parse and load uniprot xml
    parse_features: list of features to be parsed (modified residue for PTM etc.)
    '''
    pf="_"+"_".join([p.replace(" ","_") for p in parse_features])
    path_pkl = tmp_data_path/(source.stem+(pf if len(pf)>1 else "")+".pkl")
    if path_pkl.exists():
        print("Loading uniprot pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl, "not found. Parsing uniprot xml...")
        df=parse_uniprot_xml(source,parse_features=parse_features)
        df.to_pickle(path_pkl)
        return df

def load_uniref(source=path_uniref,source_uniprot=path_sprot):
    '''parse and load uniref xml
    '''
    path_pkl = tmp_data_path/(source.stem+("_"+source_uniprot.stem if source_uniprot is not None else "full")+".pkl")
    if path_pkl.exists():
        print("Loading uniref pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl,"not found. Parsing uniref...")
        df_selection = None if source_uniprot is None else load_uniprot(source_uniprot)
        df = parse_uniref(source,max_entries=0,parse_sequence=False,df_selection=df_selection)
        df.to_pickle(path_pkl)
        return df

def load_cdhit(df, cluster_type, dataset):
    '''loads/creates cluster dataframe for a given sequence dataframe (saves pickled result for later runs)
    cluster_type: cdhit05 (uniref-like cdhit down to threshold 0.5 in three stages), cdhit04 (direct cdhit down to threshold 0.4) and recalculate_cdhit05 and recalculate_cdhit04 (which do not make use of cached clusters)
    dataset: e.g. sprot determines the name of the pkl file
    '''
    path_pkl = tmp_data_path/(cluster_type+"_"+dataset+".pkl")
    
    if path_pkl.exists() and not(cluster_type == "recalculate_cdhit04" or cluster_type == "recalculate_cdhit05"):
        print("Loading cdhit pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl,"not found. Running cdhit...")
        if(cluster_type=="cdhit05" or cluster_type=="recalculate_cdhit05"):#uniref-like cdhit 0.5 in three stages
            threshold=[1.0,0.9,0.5]
            alignment_coverage=[0.0,0.9,0.8]
        else:#direct cdhit clustering to 0.4
            threshold=[0.4]
            alignment_coverage=[0.0]
        df=clusters_df_from_sequence_df(df[["sequence"]],threshold=threshold,alignment_coverage=alignment_coverage)
        df.to_pickle(path_pkl)
        return df
        
def write_log_header(path, kwargs, filename="logfile.log",append=True):
    path.mkdir(exist_ok=True)
    (path/"models").mkdir(exist_ok=True)

    if "self" in kwargs:
        del kwargs["self"]
    
    print("======================================\nCommand:"," ".join(sys.argv))
    time = datetime.datetime.now() 
    print("started at ",time)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    print("Commit:",commit)
    
    
    print("\nArguments:")
    for k in sorted(kwargs.keys()):
        print(k,":",kwargs[k])
    print("")

    filepath=path/filename
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        f.write("\n\nCommand "+" ".join(sys.argv)+"\n")
        f.write("started at "+str(time)+"\n")
        f.write("Commit "+str(commit)+"\n")
        f.write("\nArguments:\n")
        for k in sorted(kwargs.keys()):
            f.write(k+": "+str(kwargs[k])+"\n")
        f.write("\n") 

def write_log(path, text, filename="logfile.log",append=True):
    filepath=path/filename
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        f.write(str(text)+"\n")
######################################################################################################
class Preprocess(object):
    """Preprocessing class (implemented as class to allow different function calls from CLI)."""

    #############################
    # GENERAL
    #############################    
    def _preprocess_default(self,path,df,df_cluster,pad_idx=0,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,drop_fragments=False,minproteinexistence=0,exclude_aas=[],nfolds=None,sampling_ratio=[0.8,0.1,0.1],ignore_pretrained_clusters=False,save_prev_ids=True,regression=False,sequence_output=False,bpe=False,bpe_vocab_size=100, sampling_method_train=1, sampling_method_valtest=3,subsampling_ratio_train=1.0,randomize=False,random_seed=42,mask_idx=1,tok_itos_in=[],label_itos_in=[],pretrained_path=None, df_redundancy=None, df_cluster_redundancy=None):
        '''internal routine for lm preprocessing
        path: Pathlib model path
        df_cluster: clusters dataframe
        pad_idx: index of the padding token

        sequence_len_min_aas: only consider sequences of at least sequence_len_min_aas aas length 0 for no restriction
        sequence_len_max_aas: only consider sequences up to sequence_len_max_aas aas length 0 for no restriction
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization) 0 for no restriction
        drop_fragments: drop proteins with fragment marker (requires fragment column in df)
        minproteinexistence: drop proteins with proteinexistence < value (requires proteinexistence column in df)
        exclude_aas: drop sequences that contain aas in the exclude list e.g. exclude_aas=["B","J","X","Z"] for sequences with only canonical AAs ( B = D or N, J = I or L, X = unknown, Z = E or Q)

        nfolds: if None perform a single split according to sampling_ratio else nfold CV split
        sampling_ratio: sampling ratios for train/val/test
        regression: regression task
        sequence_output: output/label is a sequence
        ignore_pretrained_clusters: do not take into account existing clusters from the previous LM step during train-test-split

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        subsampling_ratio_train: artificially reduce train (clusters) to specified ratio (1.0 for full dataset)
        mask_idx: index of the mask token (for BERT masked LM training) None for none

        tok_itos_in: allows to pass tok_itos_in (used for trembl processing) will use pretrained tok_itos if an empty list is passed

        df_redundancy: optional df with same structure as df with additional redundant sequences
        df_cluster_redundancy: optional cluster df including redundant and original sequences
        '''

        #filter by fragments
        if(drop_fragments):
            if("fragment" in df.columns):
                df = df[df.fragment == 0]
                if(df_redundancy is not None):
                    df_redundancy = df_redundancy[df_redundancy.fragment == 0]
            else:
                print("Warning: could not drop fragments as fragment column is not available")
        #filter by proteinexistence
        if(minproteinexistence>0):
            if("proteinexistence" in df.columns):
                df = df[df.proteinexistence >= minproteinexistence]
                if(df_redundancy is not None):
                    df_redundancy = df_redundancy[df_redundancy.proteinexistence >= minproteinexistence]
            else:
                print("Warning: could not filter by protein existence as proteinexistence column is not available")
        #filter by non-canonical AAs
        if(len(exclude_aas)>0):
            df = filter_aas(df, exclude_aas)
            if(df_redundancy is not None):
                df_redundancy = filter_aas(df_redundancy, exclude_aas)
        #filter by aa length
        if(sequence_len_min_aas>0):
            df = df[df.sequence.apply(len)>sequence_len_min_aas]
            if(df_redundancy is not None):
                df_redundancy = df_redundancy[df_redundancy.sequence.apply(len)>sequence_len_min_aas]
        if(sequence_len_max_aas>0):
            df = df[df.sequence.apply(len)<sequence_len_max_aas]
            if(df_redundancy is not None):
                df_redundancy = df_redundancy[df_redundancy.sequence.apply(len)<sequence_len_max_aas]

        if(bpe):
            spt=sentencepiece_tokenizer(path if pretrained_path is None else pretrained_path,df,vocab_size=bpe_vocab_size)

        if(len(tok_itos_in)==0 and pretrained_path is not None):            
            tok_itos_in = np.load(pretrained_path/"tok_itos.npy")
        if(pretrained_path is not None and ignore_pretrained_clusters is False):
            if(nfolds is None):
                train_ids_prev = np.load(pretrained_path/"train_IDs_prev.npy")
                val_ids_prev = np.load(pretrained_path/"val_IDs_prev.npy")
                test_ids_prev = np.load(pretrained_path/"test_IDs_prev.npy")
            else:
                cluster_ids_prev = np.load(pretrained_path/"cluster_IDs_CV_prev.npy")
        else:
            if(nfolds is None):
                train_ids_prev = []
                val_ids_prev = []
                test_ids_prev =[]
            else:
                cluster_ids_prev = []

        prepare_dataset(path,df,tokenizer=(spt.tokenize if bpe is True else list_tokenizer),pad_idx=pad_idx,sequence_len_max_tokens=sequence_len_max_tokens,mask_idx=mask_idx,tok_itos_in=tok_itos_in,label_itos_in=label_itos_in,df_seq_redundant=df_redundancy,regression=regression,sequence_output=sequence_output)
        ids_current_all = np.load(path/"ID.npy") #filtered by sequence length
        ids_current = np.intersect1d(list(df.index),ids_current_all) #ids_current_all might contain more sequences
        ids_current_redundancy = [] if df_cluster_redundancy is None else  np.intersect1d(list(df_redundancy.index),ids_current_all)
        if(nfolds is None):
            train_test_split(path,ids_current,ids_current_all,df_cluster,train_ids_prev=train_ids_prev,val_ids_prev=val_ids_prev,test_ids_prev=test_ids_prev,sampling_ratio=sampling_ratio,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,subsampling_ratio_train=subsampling_ratio_train,randomize=randomize,random_seed=random_seed,save_prev_ids=save_prev_ids,ids_current_redundancy=ids_current_redundancy,df_cluster_redundancy=df_cluster_redundancy)
        else:
            cv_split(path,ids_current,ids_current_all,df_cluster,clusters_prev=cluster_ids_prev,nfolds=nfolds, sampling_method_train=sampling_method_train, sampling_method_valtest=sampling_method_valtest, randomize=randomize, random_seed=random_seed, save_prev_ids=save_prev_ids)

    #############################
    # LM
    ############################# 
    def lm_sprot(self, source_sprot=path_sprot,source_uniref=path_uniref,only_human_proteome=False,drop_fragments=False,minproteinexistence=0,exclude_aas=[],working_folder="./lm_sprot",pretrained_folder="",pad_idx=0,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,nfolds=None,sampling_ratio=[0.9,0.05,0.05],cluster_type="uniref",ignore_clusters=False,bpe=False,bpe_vocab_size=100,sampling_method_train=1, sampling_method_valtest=3,subsampling_ratio_train=1.0,randomize=False,random_seed=42,mask_idx=1):

        '''prepare sprot LM data
        
        only_human_proteome: filter only human proteome
        drop_fragments: drop proteins marked as fragments
        minproteinexistence: drop proteins with protein existence smaller than minproteinexistence
        working_folder: path of the data folder to be created (as string)
        pretrained_folder: path to pretrained folder (as string; empty string for none)
        pad_idx: index of the padding token
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization) 0 for no restriction
        
        nfolds: number of CV splits; None for single split
        sampling_ratio: sampling ratios for train/val/test
        ignore_clusters: do not use cluster information for train/test split
        cluster_type: source of clustering information (uniref, cdhit05: cdhit threshold 0.5 similar to uniref procedure, cdhit04: cdhit with threshold 0.4)

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        subsampling_ratio_train: portion of train clusters used
        pick_representative_for_val_test: just select a single representative per cluster for validation and test set
        mask_idx: index of the mask token (for BERT masked LM training) None for none
        '''
        print("Preparing sprot LM")

        LM_PATH=Path(working_folder)
        write_log_header(LM_PATH,locals())

        source_sprot = Path(source_sprot)

        df = load_uniprot(source=source_sprot)

        if(only_human_proteome):
            df = filter_human_proteome(df)
            print("Extracted {} human proteines".format(df.shape[0]))
        
        if(ignore_clusters is False):
            if(cluster_type[:6]=="uniref"):
                df_cluster = load_uniref(source_uniref,source_sprot)
            else:
                df_cluster = load_cdhit(df,cluster_type,source_sprot.stem)
        else:
            df_cluster = None
        self._preprocess_default(path=LM_PATH,df=df,df_cluster=df_cluster,pad_idx=pad_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,drop_fragments=drop_fragments,minproteinexistence=minproteinexistence,exclude_aas=exclude_aas,nfolds=nfolds,sampling_ratio=sampling_ratio,bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,subsampling_ratio_train=subsampling_ratio_train,randomize=randomize,random_seed=random_seed,mask_idx=mask_idx,pretrained_path=Path(pretrained_folder) if pretrained_folder !="" else None)
    
    def lm_netchop(self,working_folder="./lm_netchop",pretrained_folder="",
               existing_netchop_peptides=None, netchop_path="netchop", protein_fasta_file="./proteins.fasta",
               pad_idx=0,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,exclude_aas=[],nfolds=None,sampling_ratio=[0.9,0.05,0.05],
               netchop_min_length=8, netchop_max_length=20, netchop_repeats=30,
               cluster_type="cdhit05",ignore_clusters=True,bpe=False,bpe_vocab_size=100,
               sampling_method_train=1, sampling_method_valtest=3,randomize=False,random_seed=42,mask_idx=1):
        '''
        Prepare netchop digested sprot LM data. Starts netchop as a suprocess and slices proteins from fasta file. Tokanizes sequences and performs train-val-test split.
       
        working_folder: path of the data folder to be created (as string)
        pretrained_folder: path to pretrained folder (as string; empty string for none)
        pad_idx: index of the padding token
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization) 0 for no restriction
        
        netchop_path: e.g. ./netchop to call netchop in present directory (the default value requires to place symlink to netchop tcsh in netchop3.1 into searchpath e.g. ~/bin) BE CAREFUL NETCHOP TMP PATH MAY NOT BE TOO LONG
        protein_fasta_file: fasta file with proteins
        existing_netchop_peptides: None if not available, filename if netchop digested peptides are available as fasta file
        netchop_min_length: minimum length of netchop slices sequence, shorter sequences are discarded
        netchop_max_length: maximum length of netchop slices sequence, longer sequences are discarded
        netchop_repeats: numbers of iterations stochastic netchop slicing is applied to the whole set of proteins
        
        nfolds: number of CV splits; None for single split
        sampling_ratio: sampling ratios for train/val/test
        ignore_clusters: do not use cluster information for train/test split
        cluster_type: source of clustering information (uniref, cdhit05: cdhit threshold 0.5 similar to uniref procedure, cdhit04: cdhit with threshold 0.4)

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        pick_representative_for_val_test: just select a single representative per cluster for validation and test set
        mask_idx: index of the mask token (for BERT masked LM training) None for none
        '''
        print("Preparing netchop digested sprot LM data")

        LM_PATH=Path(working_folder)
        write_log_header(LM_PATH,locals())

        if existing_netchop_peptides is not None:
            peptides = proteins_from_fasta(data_path/existing_netchop_peptides)
            df = sequences_to_df(peptides)
        else:
            peptides = netchop_digest(protein_fasta_file, netchop_path=netchop_path, threshold=0.7, repeats=netchop_repeats, min_length=netchop_min_length, max_length=netchop_max_length,verbose=True)
            df = sequences_to_df(peptides)
        
        # TODO
        if(ignore_clusters is False):
            df_cluster = load_cdhit(df,cluster_type,"netchop_peptides")
        else:
            df_cluster = None
        
        self._preprocess_default(path=LM_PATH,df=df,df_cluster=df_cluster,pad_idx=pad_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,exclude_aas=exclude_aas,nfolds=nfolds,sampling_ratio=sampling_ratio,
         bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,
         randomize=randomize,random_seed=random_seed,mask_idx=mask_idx,pretrained_path=Path(pretrained_folder) if pretrained_folder !="" else None)

    def clas_mhc_kim(self,mhc_select,cv_type="gs",cv_fold=0,working_folder="./clas_mhc",pretrained_folder="./lm_mhc"):
        '''
        Prepares Kim14 data of one allele with BD09 as train set and Blind as test set.
        
        The dataset  is available at http://tools.iedb.org/main/datasets/ under bulletpoint "Dataset size and composition impact the reliability of performance benchmarks for peptide-MHC binding predictions." as benchmark_reliability.tar.gz and should be saved in ../data
        
        Returns raw dataframe with columns (species,mhc,peptide_length,cv,sequence,inequality,meas,label,cluster_ID) and allele ranking as dataframe with columns ("mhc","rank") if mhc_select snf working_folder are set to None.
        
        mhc_select: int between 0 and 52, choose allele by frequency rank in Binding Data 2009

        cv_type: string, strategy for 5-fold cross validation, options:
        - None: No cv-strategy, cv column is filled with 'TBD'
        - sr: removal of similar peptides seperatly in binder/ non-binder set, using similarity threshold of 80%, similarity found with 'Hobohm 1 like algorithm'
        - gs: grouping similar peptides in the same cv-partition
        - rnd: random partioning
    
        cv_fold: 0..4 selects the cv_fold to be used as validation set (in case cv_type!= None)
        '''
        if mhc_select is not None and working_folder is not None:
            print("Preparing mhc classification dataset #"+str(mhc_select)+" ...")
            CLAS_PATH = Path(working_folder)
            LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
            write_log_header(CLAS_PATH,locals())
		
        df_clas = generate_mhc_kim(cv_type=cv_type, mhc_select=mhc_select, to_csv=False, filename=None, data_dir=data_path,regression=True,transform_ic50="log")

        #generate fake cluster df for train test split
        df_clas["cluster_ID"]=0 #mark everything as train
        if(not cv_type in ["sr","gs","rnd"]):#use blind set as val set (if no cv splits provided)
            df_clas.loc[df_clas.cv=="blind","cluster_ID"]=1
        else:#cv_type specified
            df_clas.loc[df_clas.cv=="blind","cluster_ID"]=2#assign blind to test
            df_clas.loc[df_clas.cv==cv_fold,"cluster_ID"]=1#assign selected cv_fold to val

        if mhc_select is not None and working_folder is not None:
            self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_clas,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
        else:
            return df_clas,df_clas[df_clas.cv!="blind"].groupby('mhc').size().sort_values(ascending=False).reset_index()["mhc"].reset_index().rename(columns={"index":"rank"})
            
    
    def clas_mhc_flurry(self,mhc_select,working_folder="./clas_mhc_flurry",pretrained_folder="./lm_mhc_flurry",test_set=None,pad_idx=0,mask_idx=1,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,bpe=False,bpe_vocab_size=100,random_seed=42,regression=True,transform_ic50=None,label=False):
        '''
       test_set="hpv" available for alleles
       ['HLA-A*01:01','HLA-A*11:01','HLA-A*02:01','HLA-A*24:02',
       'HLA-A*03:01','HLA-B*15:01','HLA-B*07:02']
        '''
        print("Preparing mhc classification dataset "+str(mhc_select)+" ...")
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())
        
        df_clas = generate_mhc_flurry(ms='noMS', mhc_select=mhc_select, binder_threshold=500, filter_length=True, label=label,data_dir=data_path, regression=regression,transform_ic50=transform_ic50)
        
        if(test_set=="abelin"):
            df_test = generate_abelin(mhc_select=mhc_select,data_dir=data_path)
            df_clas = pd.concat([df_clas,df_test], ignore_index=True,sort=True)
        elif(test_set=="hpv"):
            flurry_hpv_map = {'HLA-A*01:01':'HLAA1',
                  'HLA-A*11:01':'HLAA11',
                  'HLA-A*02:01':'HLAA2',
                  'HLA-A*24:02':'HLAA24',
                  'HLA-A*03:01':'HLAA3',
                  'HLA-B*15:01':'HLAB15',
                  'HLA-B*07:02':'HLAB7'}
            df_test = prepare_hpv(flurry_hpv_map[mhc_select], data_dir="../git_data")
            df_test["cluster_ID"]=2
            df_clas = pd.concat([df_clas,df_test], ignore_index=True,sort=True)
 
        #self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_clas,pad_idx=pad_idx,mask_idx=mask_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=-1,sampling_method_valtest=-1,regression=regression)
        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df,df_cluster=df,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
    
    def clas_mhc_i_zhao(self, mhc_select, working_folder="./clas_mhc_i_zhao",pretrained_folder="./lm_mhc",train_set="MHCFlurry18"):
        '''
        Prepares one allele from IEDB16_I test data from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006457:
        Use either Kim14 or MHCFlurry18 data as training set.
        
        Choose the allele by frequency rank in the selected training set.
        
        Returns raw dataframe with columns ('ID', 'allele', 'cluster_ID', 'inequality', 'label', 'meas','measurement_source', 'measurement_type', 'original_allele','sequence']) and allele ranking as dataframe with columns ("allele","rank") if mhc_select snf working_folder are set to None.
        
        mhc_select: integer between 0...29 or ['HLA-A-0205', 'HLA-B-2704', 'HLA-B-2706'], the latter are not in Kim data, thus MHCFlurry18 data is used
        train_set: "MHCFlurry18" or "Kim14"
        '''
        assert train_set in ["MHCFlurry18","Kim14"], "no valid train set"
        
        if mhc_select is not None and working_folder is not None:
            print("Preparing mhc I dataset #"+str(mhc_select))
            CLAS_PATH = Path(working_folder)
            LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
            write_log_header(CLAS_PATH,locals())
        
        df_test = prepare_mhci_pcbi()
        df_test["cluster_ID"]=2
        
        if train_set=="Kim14":
            if mhc_select in ['HLA-A-0205', 'HLA-B-2704', 'HLA-B-2706']:
                df_train = generate_mhc_flurry(ms='noMS', mhc_select=None, regression=True, transform_ic50="log", data_dir="../data")
                df_train["allele"] = df_train["allele"].apply(lambda x: x.replace("*","-",1).replace(":","",1) if  x.startswith("HLA") else x)
            else:
                tmp = generate_mhc_kim(cv_type="gs", mhc_select=None, regression=True, transform_ic50="log",data_dir="../data/benchmark_mhci_reliability/binding")
                df_train = generate_mhc_kim(cv_type="gs", mhc_select=None, regression=True, transform_ic50="log",data_dir="../data/benchmark_mhci_reliability/binding").rename(columns={"mhc":"allele"})
                df_train["cluster_ID"]=0 #mark everything as train
                df_train.loc[df_train.cv==0,"cluster_ID"]=1 # validation set
                train_alleles_also_in_test_set = df_train["allele"][df_train["allele"].isin(df_test["allele"].unique())].unique()
                allele_ranking = df_train[df_train["allele"].isin(train_alleles_also_in_test_set)].groupby("allele").size().sort_values(ascending=False).index
        elif train_set=="MHCFlurry18":
            df_train = generate_mhc_flurry(ms='noMS', mhc_select=None, regression=True, transform_ic50="log", data_dir="../data")
            df_train["allele"] = df_train["allele"].apply(lambda x: x.replace("*","-",1).replace(":","",1) if  x.startswith("HLA") else x)
            test_alleles = df_test.allele.unique()
            
            df_train = df_train[df_train.allele.isin(test_alleles)]
            print(test_alleles.shape, df_train.allele.nunique())
            allele_ranking = df_train.groupby("allele").size().sort_values(ascending=False).index
       
        df = concat([df_train,df_test],ignore_index=True,sort=True)
        df.duplicated(["allele","sequence"],keep="last")
        # Take sequences also present in test set out of the training set
        df = df[~df.duplicated(["allele","sequence"],keep="last")]
        
        if mhc_select is not None:
            if train_set=="Kim14":
                if mhc_select in range(30): 
                    #print(allele_ranking[mhc_select])
                    df = df[df.allele==allele_ranking[mhc_select]]
                elif mhc_select in ['HLA-A-0205', 'HLA-B-2704', 'HLA-B-2706']:
                    df = df[df.allele==mhc_select]
            elif train_set=="MHCFlurry18":
                df = df[df.allele==allele_ranking[mhc_select]]
        
        if working_folder is not None:
            self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df,df_cluster=df,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
        else:
            return df,df_train.groupby("allele").size().sort_values(ascending=False).reset_index()["allele"].reset_index().rename(columns={"index":"rank"})
    
    def clas_mhc_i_hpv(self, mhc_select, working_folder="./clas_mhc_i_hpv",pretrained_folder="./lm_mhc", train_set="MHCFlurry18"):
        '''
        Prepare test data from Bonsack M, Hoppe S, Winter J, Tichy D, Zeller C, Küpper MD, et al. Performance evaluation of MHC class-I binding prediction tools based on an experimentally validated MHC–peptide binding data set. Cancer Immunol Res 2019;7:719–36.
        Save the dataset as HPV_data.csv in ./data.
        
        Use either Kim14 or MHCFlurry18 data as training set.
        
        mhc_select: string from ['HLAA1', 'HLAA2', 'HLAA3', 'HLAA11', 'HLAA24', 'HLAB7', 'HLAB15']
        train_set: "MHCFlurry18" or "Kim14"
        '''
        assert train_set in ["MHCFlurry18","Kim14"], "no valid train set"
                    
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        
        df = prepare_hpv(mhc_select, data_dir="../git_data")
        df["cluster_ID"] = 2 #mark everything as test
        
        if train_set=="MHCFlurry18":
            flurry_hpv_map = {'HLAA1': 'HLA-A*01:01',
                             'HLAA11': 'HLA-A*11:01',
                             'HLAA2': 'HLA-A*02:01',
                             'HLAA24': 'HLA-A*24:02',
                             'HLAA3': 'HLA-A*03:01',
                             'HLAB15': 'HLA-B*15:01',
                             'HLAB7': 'HLA-B*07:02'}
            df_train = generate_mhc_flurry(ms='noMS', mhc_select=flurry_hpv_map[mhc_select], regression=True, transform_ic50="log", binder_threshold=500, filter_length=True, label_binary=False, data_dir=data_path)
        elif train_set=="Kim14":
            kim_hpv_map = {'HLAA1': 'HLA-A-0101',
                          'HLAA11': 'HLA-A-1101',
                          'HLAA2': 'HLA-A-0201',
                          'HLAA24': 'HLA-A-2402',
                          'HLAA3': 'HLA-A-0301',
                          'HLAB15': 'HLA-B-1501',
                          'HLAB7': 'HLA-B-0702'}
            df_train = generate_mhc_kim(cv_type="gs", mhc_select=kim_hpv_map[mhc_select], regression=True, transform_ic50="log",data_dir=data_path).rename(columns={"mhc":"allele"})
            #df_train = df_train[df_train["allele"]==kim_hpv_map[mhc_select]]
            df_train["cluster_ID"]=0 #mark everything as train
            df_train.loc[df_train.cv==0,"cluster_ID"]=1#assign one cv_fold to val
            
         
        df = pd.concat([df, df_train], ignore_index=True,sort=True)

        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df,df_cluster=df,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
    def clas_mhc_sars_cov(self, mhc_select, train_on, MS=False, working_folder="./clas_mhc_i_sars_cov",pretrained_folder="./lm_mhc"):
        '''
        Prepare test data from https://www.nature.com/articles/s41598-020-77466-4#MOESM1 - stability measurements of SARS-CoV-2 peptides
        
        train_on: string, options "netmhcpan41", "netmhcpan4", "flurry"

        Use MHCFlurry18 or NetMHCpan data as training set for MHC I and NetMHCpan dataset for MHC II
        Optionally use MS data from NetMHCpan with MS set to True.
        
        mhc_select: string from ["1 A0101",
                                "2 A0201",
                                "3 A0301",
                                "4 A1101",
                                "5 A2402",
                                "6 B4001",
                                "7 C0401",
                                "8 C0701",
                                "9 C0702",
                                "10 C0102",
                                "11 DRB10401"]
        working_folder: if None, returns dataframe
        '''                    
        CLAS_PATH = Path(working_folder) if working_folder is not None else None
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        
        df = prepare_sars_cov(mhc_select, data_dir=data_path)
        df["cluster_ID"] = 2 #mark everything as test
        if MS:
            df["label"] = df["label"] >= 60
        
        if mhc_select=="11 DRB10401":
            # always use netmhc 3.2 data for training
            df_train = prepare_mhcii_netmhcpan(mhc_select="DRB1_0401", MS=MS, data_dir=data_path, netmhc_data_version="3.2")
        else:
            # mapping between sheet titles in Covid19-Intavis-Immunitrack-datasetV2.xlsx corresponding to allele names 
            # to allele names in MHCFlurry18 dataset
            if train_on=="flurry":
                allele_map = {"1 A0101": 'HLA-A*01:01',
                        "2 A0201": 'HLA-A*02:01',
                        "3 A0301": 'HLA-A*03:01',
                        "4 A1101": 'HLA-A*11:01',
                        "5 A2402": 'HLA-A*24:02',
                        "6 B4001": 'HLA-B*40:01',
                        "7 C0401": 'HLA-C*04:01',
                        "8 C0701": 'HLA-C*07:01',
                        "9 C0702": 'HLA-C*07:02' ,
                        "10 C0102": 'HLA-C*01:02'}
                df_train = generate_mhc_flurry(ms='noMS', mhc_select=allele_map[mhc_select], regression=True, transform_ic50="log", binder_threshold=500, filter_length=True, label_binary=False, data_dir=data_path)
            else:
                allele_map = {"1 A0101": 'HLA-A01:01',
                        "2 A0201": 'HLA-A02:01',
                        "3 A0301": 'HLA-A03:01',
                        "4 A1101": 'HLA-A11:01',
                        "5 A2402": 'HLA-A24:02',
                        "6 B4001": 'HLA-B40:01',
                        "7 C0401": 'HLA-C04:01',
                        "8 C0701": 'HLA-C07:01',
                        "9 C0702": 'HLA-C07:02' ,
                        "10 C0102": 'HLA-C01:02'}
                netmhc_version = "4.0" if train_on=="netmhcpan4" else "4.1"
                df_train = prepare_mhci_netmhcpan_4(mhc_select=allele_map[mhc_select], MS=MS, data_dir=data_path, netmhc_data_version=netmhc_version)


        df = pd.concat([df, df_train], ignore_index=True,sort=True)

        if MS:
            df["label"] = df["label"].astype(int)
        
        if working_folder is not None:
            self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df,df_cluster=df,sampling_method_train=-1,sampling_method_valtest=-1,regression=False if MS else True,ignore_pretrained_clusters=True)
        else:
            return df

    def clas_mhc_sars_cov_pan(self, mhc_class, working_folder="./clas_mhc_i_sars_cov_pan"):
        
        CLAS_PATH = Path(working_folder)

        train = prepare_mhci_netmhcpan_4(None, with_MHC_seq=True, data_dir=data_path, netmhc_data_version="4.0")
        test = prepare_sars_cov(None, mhc_class=mhc_class, with_MHC_seq=True, data_dir=data_path)
        test["cluster_ID"] = 2

        df = pd.concat([train,test], axis=0, sort=True, ignore_index=True)
        
        # concat MHC pseudo sequence and peptide sequence
        df["sequence"] = df["sequence1"] + "x" + df["sequence"]  

        prep = Preprocess()

        # provide previous tokens so that "x" is mapped to '_pad_'
        prev_tok = ['_pad_', '_mask_', '_bos_', 'G', 'E', 'S', 'P', 'A', 'D', 'T', 'V', 'L', 'R', 'N', 'I', 'Q', 'K', 'C', 'F', 'H', 'M', 'Y', 'W']

        self._preprocess_default(path=CLAS_PATH,
                                df=df,df_cluster=df,
                                sampling_method_train=-1,sampling_method_valtest=-1,
                                regression=True,ignore_pretrained_clusters=True,
                                tok_itos_in=prev_tok)



                    
    def clas_mhc_ii(self, mhc_select, working_folder="./clas_mhc_ii",pretrained_folder="./lm_mhc"):  
        '''
        Prepares IEDB16_II data from https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006457: of on allele.
        Uses Wang10 datset (classII_binding_data_Nov_16_2009.zip from http://tools.iedb.org/mhcii/download/) as training data.
        
        mhc_select: integer between 0...24
        '''
        print("Preparing mhc II dataset #"+str(mhc_select)+" ...")
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())
        
        df_train = prepare_mhcii_iedb2009()
        df_test = prepare_mhcii_pcbi()
        
        # select only one train/test split out of the 5
        df_train = df_train[df_train["cv_fold"]=="0"]
        
        # select one allele based on training set size ranking
        train_alleles_also_in_test_set = df_train["allele"][df_train["allele"].isin(df_test["allele"].unique())].unique()
        allele_ranking = df_train[df_train["allele"].isin(train_alleles_also_in_test_set)].groupby("allele").size().sort_values(ascending=False).index
        if mhc_select is not None:
            df_train = df_train[df_train["allele"]==allele_ranking[mhc_select]]
            df_test = df_test[df_test["allele"]==allele_ranking[mhc_select]]
        
        # use the "test" part of the selected cv fold as validation set
        df_train["cluster_ID"] = 0
        df_train.loc[df_train.traintest=="test","cluster_ID"]=1
        # use the pcbi data as test set
        df_test["cluster_ID"] = 2
        df_clas = concat([df_train,df_test], ignore_index=True,sort=True)
        
        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_clas,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
            
    def clas_mhc_ii_iedb2016(self, mhc_select, cv_fold, working_folder="./clas_mhc_ii_iedb2016",pretrained_folder="./lm_mhc"):
        '''
        prepares mhc ii data from http://www.cbs.dtu.dk/suppl/immunology/NetMHCIIpan-3.2/
        mhc_select: integer between 0...60
        cv_fold: integer between 0 ...4
        '''
        print("Preparing mhc II dataset #"+str(mhc_select)+ " fold #" + str(cv_fold) + " ...")
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())
        
        df = prepare_mhcii_iedb2016(mhc_select, cv_fold, path_iedb="../data/iedb2016", path_jensen_csv="../data/jensen_et_al_2018_immunology_supplTabl3.csv")
        
        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df,df_cluster=df,sampling_method_train=-1,sampling_method_valtest=-1,regression=True,ignore_pretrained_clusters=True)
    
if __name__ == '__main__':
    fire.Fire(Preprocess)
