import os, sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau

from proteomics_preprocessing import Preprocess
from utils.proteomics_utils import  prepare_mhcii_iedb2009, prepare_mhcii_pcbi

##### Collect predictions during training #####
def collect_preds_npz(data_dir, n_alleles, filename=None, 
                      subfoldername="allele", ensemble_i=None, preds_filename='preds_test.npz', 
                      ranked_alleles=True,allele_list=None):
    """
    store predictions per ensemble and model parameter in npz file during ensemble training
    data_dir: string, directory containing n_alleles allele folders
    filename: to store npz file, if None, the model_filename_prefix will be used
   """
    working_path = Path(data_dir)
    
    if(ranked_alleles):
        allele_list = np.arange(n_alleles).astype(str)
    
    #load predictions of first allele
    clas_path = working_path/(subfoldername+allele_list[0])

    test_preds = np.load(clas_path/preds_filename, allow_pickle=True)
    # load model parameter
    with open(clas_path/"kwargs.pkl", "rb") as f:
            args_0 = pickle.load(f)
    del args_0["working_folder"]
    
    results = []
    result = pd.DataFrame(data={'preds':np.squeeze(test_preds['preds']), 'targs':test_preds['targs'],'ID':test_preds['IDs']})
    result['rank'] = allele_list[0]
    results.append(result)
    
    # load predictions and parameter for all other alleles
    for allele in allele_list[1:]:     
        clas_path = working_path/(subfoldername+allele)
        test_preds = np.load(clas_path/preds_filename, allow_pickle=True)
        with open(clas_path/"kwargs.pkl", "rb") as f:
            args = pickle.load(f)
        del args["working_folder"]
        
        # only append if the model parameters match the ones from the first allele
        if args==args_0:
            result = pd.DataFrame(data={'preds':np.squeeze(test_preds['preds']), 'targs':test_preds['targs'],
                                        'ID':test_preds['IDs']})
            result['rank'] = allele
            results.append(result)
        else:
            print("Model parameter did not match, results not included.")

    
    results = pd.concat(results, ignore_index=True)
    
    # store
    if filename is None:
        filename = args_0["model_filename_prefix"]
    
    if ensemble_i is not None:
        np.savez(working_path/(filename+".npz"), 
        IDs=results["ID"].values,rank=results["rank"].values,
        preds=results["preds"].values, targs=results["targs"].values, 
        arg_keys=list(args_0.keys()), args=list(args_0.values()),
        ensemble_i=ensemble_i
       )
    else:
        np.savez(working_path/(filename+".npz"), 
                IDs=results["ID"].values,rank=results["rank"].values,
                preds=results["preds"].values, targs=results["targs"].values, 
                arg_keys=list(args_0.keys()), args=list(args_0.values())
               )

##### Load Predictions #####
def inequality_from_targ(x):
    if x<=1:
        return "="
    elif x>=2 and x<=3:
        return ">"
    elif x>=4 and x<=5:
        return "<"
    else:
        return np.nan


def read_npz(file, sigmoid=True, ensemble=False, with_args=True, rank=True, transform_targs=True, dropout_test=False):
    '''
    Read the *.npz files generated by collect_preds_npz()
    '''
    res_npz = np.load(file, allow_pickle=True)

    if rank:
        df = pd.DataFrame(data={'preds':np.squeeze(res_npz['preds']), 'targs':res_npz['targs'],'ID':res_npz['IDs'], 'rank':res_npz['rank']})
    else:
        df = pd.DataFrame(data={'preds':np.squeeze(res_npz['preds']), 'targs':res_npz['targs'],'ID':res_npz['IDs'], 'allele':res_npz['allele']})
    
    if transform_targs:
        df["inequality"] = df["targs"].map(inequality_from_targ)
        df["targs"] = df["targs"].apply(back_trafo)
        df["targs"] = df["targs"].apply(to_ic50)

    if sigmoid:
        df["preds"] = df["preds"].apply(lambda x: 1/(1+np.exp(-x)))
    df["preds"] = df["preds"].apply(to_ic50)
    
    if with_args:
        df_args = pd.DataFrame(index=res_npz["arg_keys"], data={"param":res_npz["args"]})
        df = pd.concat([df, pd.concat(df.shape[0]*[df_args.T], ignore_index=True)], axis=1)
        df.rename(columns={"model_filename_prefix":"model"},inplace=True)
    else:
        df["model"] = file.stem
    
    if dropout_test:
        df["dropout"] = res_npz["dropout"]
        
    if ensemble:
        df["ensemble_i"] = res_npz["ensemble_i"]
    
    return df
 
def load_tokenized_sequences(allele_dir, allele):
    allele_path = Path(allele_dir)
    ID = np.load(allele_path/"ID.npy")
    tok = np.load(allele_path/"tok.npy",allow_pickle=True)
    tok_itos = np.load(allele_path/"tok_itos.npy")
    test_IDs = np.load(allele_path/"test_IDs.npy")
    tok = tok[test_IDs]
    sequence = [tok_itos[tok_i[1:]] for tok_i in tok] #not include bos token
    for i,seq in enumerate(sequence):
        seq_concat = ""
        for aa in seq:
            seq_concat += aa
        sequence[i] = seq_concat
    return pd.DataFrame({"sequence":sequence, "ID":test_IDs, "rank":allele})

def load_ranking(dataset):
    if dataset=="IEDB16_I":
        prep = Preprocess()
        _, ranking = prep.clas_mhc_i_zhao(mhc_select=None, working_folder=None,train_set="MHCFlurry18")
    elif dataset=="Kim14":
        kim_data_dir = Path("../data/benchmark_mhci_reliability/binding")
        kim_train_raw = pd.read_csv(kim_data_dir/"bd2009.1"/"bdata.2009.mhci.public.1.cv_gs.txt", sep='\t')
        kim_test_raw = pd.read_csv(kim_data_dir/"blind.1"/"bdata.2013.mhci.public.blind.1.txt", sep='\t')
        ranking = kim_train_raw.groupby("mhc").size().sort_values(ascending=False).iloc[:53]
        ranking = ranking.reset_index().reset_index().rename(columns={"index":"rank","mhc":"allele"}).drop(0,axis=1)
    elif dataset=="IEDB16_II":
        df_train = prepare_mhcii_iedb2009()
        df_test = prepare_mhcii_pcbi()
        # select only one train/test split out of the 5
        df_train = df_train[df_train["cv_fold"]=="0"]
        # select one allele based on training set size ranking
        train_alleles_also_in_test_set = df_train["allele"][df_train["allele"].isin(df_test["allele"].unique())].unique()
        ranking = df_train[df_train["allele"].isin(train_alleles_also_in_test_set)].groupby("allele").size().sort_values(ascending=False)
        ranking = ranking.reset_index().reset_index().rename(columns={"index":"rank","mhc":"allele"}).drop(0,axis=1)
    return ranking
    
 ##### Helper functions #####
def allele_starformat_3(x):
    if x.startswith('HLA'):
        return x[:5] + '*' + x[6:8] + ':' + x[8:]
    elif x.startswith('Mamu'):
        return x[:6] + '*'  + x[7:]
    elif x.startswith('H-2'):
        return x[:-1] + x[-1].lower()
    else:
        return x
    
def allele_starformat_flurry(x, train_data=True):
    if train_data:
        if x.startswith('HLA'):
            return x[:5] + '*' + x[6:8] + ':' + x[8:]
        elif x.startswith('Mamu'):
            return x[:6] + '*'  + x[7:] + ":01"
        elif x.startswith('H-2'):
            return x[:-1] + x[-1].lower()
        else:
            return x
    else: 
        if x.startswith('Mamu'):
            return x + ":01"
        else:
            return x

def back_trafo(x, cap=1):
    if((x>=(2*cap)) & (x<=(3*cap))):      
        return x-(2*cap)
    elif x>=(4*cap):
        return x-(4*cap)
    else:
        return x
    
def from_ic50(ic50, max_ic50=50000.0):
    """
    Convert ic50s to regression targets in the range [0.0, 1.0].
    Parameters
    ----------
    ic50 : numpy.array of float
    Returns
    --------
    numpy.array of float
    """
    x = 1.0 - (np.log10(ic50) / np.log10(max_ic50))

    return np.minimum(1.0,np.maximum(0.0, x))

def to_ic50(x, max_ic50=50000.0):

    """
    Convert regression targets in the range [0.0, 1.0] to ic50s in the range
    [0, 50000.0].
    Parameters
    ----------
    x : numpy.array of float
    Returns
    -------
    numpy.array of float
    """
    return max_ic50 ** (1.0-x) 

    
##### Evaluation Metrics ##### 
def aucroc_ic50(df,threshold=500):
    """
    Compute AUC ROC for predictions and targets in DataFrame, based on a given threshold
    Parameters
    ----------
    df : pandas.DataFrame with predictons in column "preds" and targets in column "targs" in nM
    threshold: float, binding affinty threshold for binders in nM
    Returns
    --------
    numpy.nan or float
    """
    df =df[~df["preds"].isnull()]
    is_binder =  df["targs"] >= threshold
    if is_binder.mean()==1.0 or is_binder.mean()==0.0 or np.isnan(is_binder.mean()):
        return np.nan
    else:
        return roc_auc_score(1.0*is_binder,df["preds"])  

def aucroc_hpv(df):
    #print(df["preds"].isnull().sum())
    df =df[~df["preds"].isnull()]

    if df["targs"].isnull().mean()==0 or df["targs"].isnull().mean()==1 or df.shape[0]==0:
        return np.nan
    else:
        return roc_auc_score( 1.0*(df["targs"].isnull()),df["preds"])
        
def kendalltau_eval(preds_targs, preds_col="preds", targs_col="targs"):
    preds_targs = preds_targs[["preds","targs"]]
    preds_targs = preds_targs[~preds_targs.isnull().any(axis=1)]
    preds_targs_rank = preds_targs.rank(method="average")
    return kendalltau(preds_targs_rank[preds_col].values, preds_targs_rank[targs_col].values)[0]  
    
def spearmanr_eval(preds_targs, preds_col="preds", targs_col="targs", filter_quantitative=True):
    if(filter_quantitative):
        preds_targs = preds_targs[preds_targs["inequality"] == "="]
    preds_targs = preds_targs[["preds","targs"]]
    preds_targs = preds_targs[~preds_targs.isnull().any(axis=1)]

    return preds_targs.corr(method="spearman")["preds"].loc["targs"]

##### Bootstrapping #####
def bootstrap_confidence_interval(df, statistics, iterations, alpha=0.95, 
                                  groupby_level=[0], 
                                  allele_drop={"AUC ROC":["HLA-B-2704","HLA-B-1503"],"SRCC":["HLA-B-1503"]},
                                  metric_kwargs={}):

    stat_bootstrap_mean = {stat:{} for stat in statistics}
    stat_bootstrap_overall = {stat:{} for stat in statistics}
    stat_bootstrap_aw = {stat:{} for stat in statistics}
    
    for i in range(iterations):
        if i%10==0:
            print(i)
        tmp = df.groupby(level=groupby_level).apply(pd.DataFrame.sample, frac=1, replace=True).reset_index(level=groupby_level, drop=True)
        

        for stat in statistics:

            if stat in allele_drop.keys():
                tmp_drop = tmp[~tmp["allele"].isin(allele_drop[stat])]
            else:
                tmp_drop = deepcopy(tmp)
            
            if stat in metric_kwargs.keys():
                kwargs = metric_kwargs[stat]
            else: 
                kwargs = {}
            
            # mean: compute metric for each allele and then average over alleles
            allelewise_stat_ji = tmp_drop.groupby(level=groupby_level+[groupby_level[-1]+1]).apply(statistics[stat], **kwargs)
            #print(allelewise_stat_ji.groupby(level=groupby_level).mean())
            stat_bootstrap_mean[stat][i] = allelewise_stat_ji.groupby(level=groupby_level).mean()
            
            # allewise stat
            stat_bootstrap_aw[stat][i] = allelewise_stat_ji
            
            # overall: compute metric for peptides of all alleles at once
            stat_bootstrap_overall[stat][i] = tmp.groupby(level=groupby_level[0]).apply(statistics[stat], **kwargs)

    stat_dfs_mean = {}
    stat_dfs_overall = {}
    stat_dfs_aw = {}

    
    for stat in statistics:
        lower_mean, upper_mean, mean_mean = confidence_interval(pd.concat(stat_bootstrap_mean[stat],axis=1), alpha)
        lower_overall, upper_overall, mean_overall = confidence_interval(pd.concat(stat_bootstrap_overall[stat],axis=1), 
                                                                         alpha)
        lower_aw, upper_aw, mean_aw = confidence_interval(pd.concat(stat_bootstrap_aw[stat],axis=1), 
                                                                         alpha)
        
        stat_dfs_mean[stat] = pd.DataFrame({"lower":lower_mean, "upper":upper_mean,
                                            "bs_stat_mean":mean_mean})                                                                                                    
        stat_dfs_overall[stat] = pd.DataFrame({"lower":lower_overall, "upper":upper_overall, 
                                               "bs_stat_mean":mean_overall})  
        stat_dfs_aw[stat] = pd.DataFrame({"lower":lower_aw, "upper":upper_aw, 
                                       "bs_stat_mean":mean_aw})  
    

    return {"mean":stat_dfs_mean, "overall":stat_dfs_overall, "allelewise":stat_dfs_aw}, stat_bootstrap_mean

def confidence_interval(df, alpha):
    p = ((1.0-alpha)/2.0)
    lower = df.quantile(p,axis=1)
    p = (alpha+((1.0-alpha)/2.0))
    upper = df.quantile(p,axis=1)
    return lower, upper, df.mean(axis=1)

def single_conservative_avg(df, with_error=True, groupby_keys=["metric","model"]):
    is_single = df.model.apply(lambda x: x[-5:]).isin(["ens_"+str(i) for i in range(10)])
    single = df[is_single]
    rest = df[~is_single]
    single["model"] = df.model.apply(lambda x: x[:-6]+"_sng")
    if(with_error):
        single["error"] = pd.concat((single["bs_stat_mean"]-single["lower"], single["upper"]-single["bs_stat_mean"]),axis=1).max(axis=1)
        single = single.groupby(groupby_keys).agg({"bs_stat_mean":"mean","error":"max"})
        single["upper"] = single["bs_stat_mean"]+single["error"]
        single["lower"] = single["bs_stat_mean"]-single["error"]
        del single["error"]
    else:
        single = single.groupby(groupby_keys).mean()
    return pd.concat((single.reset_index(),rest),ignore_index=True)
    
##### Details on dataset #####
def get_dataset_specs(which, n_alleles=53, prefix = "reg_mhc_kim_",data_dir = "reg_mhc_kim_human_sc3_netchop",
                      filter_len=None, filter_train=False, allele_list=None, test_is_val=False, binder_threshold=500):
    """
    which: "train" or "test"
    filter_train: if True, filter train set for alleles that include peptides of length filter_len
    """
    df_seq_test = []
    df_seq_train = []

    if allele_list is None:
        allele_list = range(n_alleles)
    for allele in allele_list:
        ID = np.load("./{}/{}{}/ID.npy".format(data_dir,prefix,allele))
        tok = np.load("./{}/{}{}/tok.npy".format(data_dir,prefix,allele))
        tok_itos = np.load("./{}/{}{}/tok_itos.npy".format(data_dir,prefix,allele))
        if(test_is_val):
            test_IDs = np.load("./{}/{}{}/val_IDs.npy".format(data_dir,prefix,allele))
            train_IDs = np.load("./{}/{}{}/train_IDs.npy".format(data_dir,prefix,allele))
        else:
            test_IDs = np.load("./{}/{}{}/test_IDs.npy".format(data_dir,prefix,allele))
            val_IDs =  np.load("./{}/{}{}/val_IDs.npy".format(data_dir,prefix,allele))
            train_IDs = np.load("./{}/{}{}/train_IDs.npy".format(data_dir,prefix,allele))
            train_IDs = np.concatenate((train_IDs, val_IDs),axis=0)
        
        label = np.load("./{}/{}{}/label.npy".format(data_dir,prefix,allele))
        inequality = np.array([inequality_from_targ(float(x)) for x in label])
        label = np.array([back_trafo(float(x)) for x in label])
        label = to_ic50(label)

        #print(test_IDs,test_IDs.shape)
        testtok = tok[test_IDs]
        traintok = tok[train_IDs]

        sequence = [tok_itos[tok_i[1:]] for tok_i in testtok] #not include bos token
        for i,seq in enumerate(sequence):
            seq_concat = ""
            for aa in seq:
                seq_concat += aa
            sequence[i] = seq_concat
        df_test = pd.DataFrame({"sequence":sequence, "ID":test_IDs, "rank":allele,
                                "label":label[test_IDs], "inequality":inequality[test_IDs]})

        sequence = [tok_itos[tok_i[1:]] for tok_i in traintok] #not include bos token
        for i,seq in enumerate(sequence):
            seq_concat = ""
            for aa in seq:
                seq_concat += aa
            sequence[i] = seq_concat
        df_train = pd.DataFrame({"sequence":sequence, "ID":train_IDs, "rank":allele,
                                 "label":label[train_IDs], "inequality":inequality[train_IDs]})

        df_seq_test.append(df_test)
        df_seq_train.append(df_train)

    df_seq_test = pd.concat(df_seq_test)
    df_seq_train = pd.concat(df_seq_train)

    df_seq_test["cluster"] = "test"
    df_seq_train["cluster"] = "train"

    df_seq = pd.concat([df_seq_train,df_seq_test],ignore_index=True)

    df_seq["mer"] = df_seq["sequence"].apply(len)
    
    if filter_len is not None:
        if(filter_train):
            test = df_seq[df_seq["cluster"]=="test"]
            test = test[test["mer"]==filter_len]
            df_seq = df_seq[df_seq["rank"].isin(test["rank"].unique())]
        else:
            df_seq = df_seq[df_seq["mer"]==filter_len]
    
    df_seq = df_seq[df_seq["cluster"]==which]
        
    size = df_seq.groupby("rank").size()

    binder_share = (df_seq["label"]<=500).mean()
    binder_share_by_rank = df_seq.groupby("rank").apply(lambda x: (x["label"]<=500).mean())


    return     {"total size":size.sum(), 
               "size per rank": size.sum(), 
               "median size": size.median(),
               "quantitative share": (df_seq["inequality"]=="=").mean(),
               "IQR":size.quantile(0.75) - size.quantile(0.25),
               "peptide lenghts":df_seq["mer"].unique(),
               "number of alleles":df_seq["rank"].nunique(),
               "binder share":binder_share, 
               "binder share per rank":binder_share_by_rank}