from lxml import etree
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

import Bio
from Bio import SeqIO


from pathlib import Path

#console
from tqdm import tqdm as tqdm
import re
import os
import itertools
#jupyter
#from tqdm import tqdm_notebook as tqdm
#not supported in current tqdm version
#from tqdm.autonotebook import tqdm

#import logging
#logging.getLogger('proteomics_utils').addHandler(logging.NullHandler())
#logger=logging.getLogger('proteomics_utils')

#for cd-hit
import subprocess

from sklearn.metrics import f1_score

import hashlib #for mhcii datasets

from utils.dataset_utils import split_clusters_single,pick_all_members_from_clusters

#######################################################################################################
#Parsing all sorts of protein data
#######################################################################################################
def parse_uniprot_xml(filename,max_entries=0,parse_features=[]):
    '''parse uniprot xml file, which contains the full uniprot information (e.g. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz)
    using custom low-level https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    c.f. for full format https://www.uniprot.org/docs/uniprot.xsd

    parse_features: a list of strings specifying the kind of features to be parsed such as "modified residue" for phosphorylation sites etc. (see https://www.uniprot.org/help/mod_res)
        (see the xsd file for all possible entries)
    '''
    context = etree.iterparse(str(filename), events=["end"], tag="{http://uniprot.org/uniprot}entry")
    context = iter(context)
    rows =[]
    
    for _, elem in tqdm(context):
        parse_func_uniprot(elem,rows,parse_features=parse_features)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)

    return df

def parse_func_uniprot(elem, rows, parse_features=[]):
    '''extracting a single record from uniprot xml'''
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    sequence=""
    #print(seqs)
    for s in seqs:
        sequence=s.text
        #print("sequence",sequence)
        if sequence =="" or str(sequence)=="None":
            continue
        else:
            break
    #Sequence & fragment
    sequence=""
    fragment_map = {"single":1, "multiple":2}
    fragment = 0
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    for s in seqs:
        if 'fragment' in s.attrib:
            fragment = fragment_map[s.attrib["fragment"]]
        sequence=s.text
        if sequence != "":
            break
    #print("sequence:",sequence)
    #print("fragment:",fragment)

    #dataset
    dataset=elem.attrib["dataset"]

    #accession
    accession = ""
    accessions = elem.findall("{http://uniprot.org/uniprot}accession")
    for a in accessions:
        accession=a.text
        if accession !="":#primary accession! https://www.uniprot.org/help/accession_numbers!!!
            break
    #print("accession",accession)

    #protein existence (PE in plain text)
    proteinexistence_map = {"evidence at protein level":5,"evidence at transcript level":4,"inferred from homology":3,"predicted":2,"uncertain":1}
    proteinexistence = -1
    accessions = elem.findall("{http://uniprot.org/uniprot}proteinExistence")
    for a in accessions:
        proteinexistence=proteinexistence_map[a.attrib["type"]]
        break
    #print("protein existence",proteinexistence)

    #name
    name = ""
    names = elem.findall("{http://uniprot.org/uniprot}name")
    for n in names:
        name=n.text
        break
    #print("name",name)

    #organism
    organism = ""
    organisms = elem.findall("{http://uniprot.org/uniprot}organism")
    for s in organisms:
        s1=s.findall("{http://uniprot.org/uniprot}name")
        for s2 in s1:
            if(s2.attrib["type"]=='scientific'):
                organism=s2.text
                break
        if organism !="":
            break
    #print("organism",organism)

    #dbReference: PMP,GO,Pfam, EC
    ids = elem.findall("{http://uniprot.org/uniprot}dbReference")
    pfams = []
    gos =[]
    ecs = []
    pdbs =[]
    for i in ids:
        #print(i.attrib["id"],i.attrib["type"])

        #cf. http://geneontology.org/external2go/uniprotkb_kw2go for Uniprot Keyword<->GO mapping
        #http://geneontology.org/ontology/go-basic.obo for List of go terms
        #https://www.uniprot.org/help/keywords_vs_go keywords vs. go
        if(i.attrib["type"]=="GO"):
            tmp1 = i.attrib["id"]
            for i2 in i:
                if i2.attrib["type"]=="evidence":
                    tmp2= i2.attrib["value"]
            gos.append([int(tmp1[3:]),int(tmp2[4:])]) #first value is go code, second eco evidence ID (see mapping below)
        elif(i.attrib["type"]=="Pfam"):
            pfams.append(i.attrib["id"])
        elif(i.attrib["type"]=="EC"):
            ecs.append(i.attrib["id"])
        elif(i.attrib["type"]=="PDB"):
            pdbs.append(i.attrib["id"])
    #print("PMP: ", pmp)
    #print("GOs:",gos)
    #print("Pfams:",pfam)
    #print("ECs:",ecs)
    #print("PDBs:",pdbs)


    #keyword
    keywords = elem.findall("{http://uniprot.org/uniprot}keyword")
    keywords_lst = []
    #print(keywords)
    for k in keywords:
        keywords_lst.append(int(k.attrib["id"][-4:]))#remove the KW-
    #print("keywords: ",keywords_lst)

    #comments = elem.findall("{http://uniprot.org/uniprot}comment")
    #comments_lst=[]
    ##print(comments)
    #for c in comments:
    #    if(c.attrib["type"]=="function"):
    #        for c1 in c:
    #            comments_lst.append(c1.text)
    #print("function: ",comments_lst)

    #ptm etc
    if len(parse_features)>0:
        ptms=[]
        features = elem.findall("{http://uniprot.org/uniprot}feature")
        for f in features:
            if(f.attrib["type"] in parse_features):#only add features of the requested type
                locs=[]
                for l in f[0]:
                    locs.append(int(l.attrib["position"]))
                ptms.append([f.attrib["type"],f.attrib["description"] if 'description' in f.attrib else "NaN",locs, f.attrib['evidence'] if 'evidence' in f.attrib else "NaN"])
        #print(ptms)

    data_dict={"ID": accession, "name": name, "dataset":dataset, "proteinexistence":proteinexistence, "fragment":fragment, "organism":organism, "ecs": ecs, "pdbs": pdbs, "pfams" : pfams, "keywords": keywords_lst, "gos": gos,  "sequence": sequence}

    if len(parse_features)>0:
        data_dict["features"]=ptms
    #print("all children:")
    #for c in elem:
    #    print(c)
    #    print(c.tag)
    #    print(c.attrib)
    rows.append(data_dict)

def parse_uniprot_seqio(filename,max_entries=0):
    '''parse uniprot xml file using the SeqIO parser (smaller functionality e.g. does not extract evidence codes for GO)'''
    sprot = SeqIO.parse(filename, "uniprot-xml")
    rows = []
    for p in tqdm(sprot):
        accession = str(p.name)
        name = str(p.id)
        dataset = str(p.annotations['dataset'])
        organism = str(p.annotations['organism'])
        ecs, pdbs, pfams, gos = [],[],[],[]
        
        for ref in p.dbxrefs:
            k = ref.split(':')
            if k[0] == 'GO':
                gos.append(':'.join(k[1:]))
            elif k[0] == 'Pfam':
                pfams.append(k[1])
            elif k[0] == 'EC':
                ecs.append(k[1])
            elif k[0] == 'PDB':
                pdbs.append(k[1:])
        if 'keywords' in p.annotations.keys():
            keywords = p.annotations['keywords']
        else:
            keywords = []
        
        sequence = str(p.seq)
        row = {
            'ID': accession, 
            'name':name, 
            'dataset':dataset,
            'organism':organism,
            'ecs':ecs,
            'pdbs':pdbs,
            'pfams':pfams,
            'keywords':keywords,
            'gos':gos,
            'sequence':sequence}
        rows.append(row)
        if(max_entries>0 and len(rows)==max_entries):
            break          

    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)

    return df
    
def filter_human_proteome(df_sprot):
    '''extracts human proteome from swissprot proteines in DataFrame with column organism '''
    is_Human = np.char.find(df_sprot.organism.values.astype(str), "Human") !=-1
    is_human = np.char.find(df_sprot.organism.values.astype(str), "human") !=-1
    is_sapiens = np.char.find(df_sprot.organism.values.astype(str), "sapiens") !=-1
    is_Sapiens = np.char.find(df_sprot.organism.values.astype(str), "Sapiens") !=-1
    return df_sprot[is_Human|is_human|is_sapiens|is_Sapiens]

def filter_aas(df, exclude_aas=["B","J","X","Z"]):
    '''excludes sequences containing exclude_aas: B = D or N, J = I or L, X = unknown, Z = E or Q'''
    return df[~df.sequence.apply(lambda x: any([e in x for e in exclude_aas]))]
######################################################################################################
def explode_clusters_df(df_cluster):
    '''aux. function to convert cluster dataframe from one row per cluster to one row per ID'''
    df=df_cluster.reset_index(level=0)
    rows = []
    if('repr_accession' in df.columns):#include representative if it exists
        _ = df.apply(lambda row: [rows.append([nn,row['entry_id'], row['repr_accession']==nn ]) for nn in row.members], axis=1)
        df_exploded = pd.DataFrame(rows, columns=['ID',"cluster_ID","representative"]).set_index(['ID'])
    else:
        _ = df.apply(lambda row: [rows.append([nn,row['entry_id']]) for nn in row.members], axis=1)
        df_exploded = pd.DataFrame(rows, columns=['ID',"cluster_ID"]).set_index(['ID'])
    return df_exploded

def parse_uniref(filename,max_entries=0,parse_sequence=False, df_selection=None, exploded=True):
    '''parse uniref (clustered sequences) xml ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz unzipped 100GB file
    using custom low-level parser https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    max_entries: only return first max_entries entries (0=all)
    parse_sequences: return also representative sequence
    df_selection: only include entries with accessions that are present in df_selection.index (None keeps all records)
    exploded: return one row per ID instead of one row per cluster
    c.f. for full format ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/README
    '''
    #issue with long texts https://stackoverflow.com/questions/30577796/etree-incomplete-child-text
    #wait for end rather than start tag
    context = etree.iterparse(str(filename), events=["end"], tag="{http://uniprot.org/uniref}entry")
    context = iter(context)
    rows =[]
    
    for _, elem in tqdm(context):
        parse_func_uniref(elem,rows,parse_sequence=parse_sequence, df_selection=df_selection)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("entry_id")
    df["num_members"]=df.members.apply(len)

    if(exploded):
        return explode_clusters_df(df)
    return df

def parse_func_uniref(elem, rows, parse_sequence=False, df_selection=None):
    '''extract a single uniref entry'''
    #entry ID
    entry_id = elem.attrib["id"]
    #print("cluster id",entry_id)
    
    #name
    name = ""
    names = elem.findall("{http://uniprot.org/uniref}name")
    for n in names:
        name=n.text[9:]
        break
    #print("cluster name",name)
    
    members=[]
    #representative member
    repr_accession = ""
    repr_sequence =""
    repr = elem.findall("{http://uniprot.org/uniref}representativeMember")
    for r in repr:
        s1=r.findall("{http://uniprot.org/uniref}dbReference")
        for s2 in s1:
            for s3 in s2:
                if s3.attrib["type"]=="UniProtKB accession":
                    if(repr_accession == ""):
                        repr_accession = s3.attrib["value"]#pick primary accession
                    members.append(s3.attrib["value"])
        if parse_sequence is True:
            s1=r.findall("{http://uniprot.org/uniref}sequence")
            for s2 in s1:
                repr_sequence = s2.text
                if repr_sequence !="":
                    break
        
    #print("representative member accession:",repr_accession)
    #print("representative member sequence:",repr_sequence)
    
    #all members
    repr = elem.findall("{http://uniprot.org/uniref}member")
    for r in repr:
        s1=r.findall("{http://uniprot.org/uniref}dbReference")
        for s2 in s1:
            for s3 in s2:
                if s3.attrib["type"]=="UniProtKB accession":
                    members.append(s3.attrib["value"]) #add primary and secondary accessions
    #print("members", members)

    if(not(df_selection is None)): #apply selection filter
        members = [y for y in members if y in df_selection.index]
    #print("all children")
    #for c in elem:
    #    print(c)
    #    print(c.tag)
    #    print(c.attrib)
    if(len(members)>0):
        data_dict={"entry_id": entry_id, "name": name, "repr_accession":repr_accession, "members":members}
        if parse_sequence is True:
            data_dict["repr_sequence"]=repr_sequence
        rows.append(data_dict)
###########################################################################################################################
#proteins and peptides from fasta
###########################################################################################################################
def parse_uniprot_fasta(fasta_path, max_entries=0):
    '''parse uniprot from fasta file (which contains less information than the corresponding xml but is also much smaller e.g. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta)'''
    rows=[]
    dataset_dict={"sp":"Swiss-Prot","tr":"TrEMBL"}
    for seq_record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        sid=seq_record.id.split("|")
        accession = sid[1]
        dataset = dataset_dict[sid[0]]
        name = sid[2]
        description = seq_record.description
        sequence=str(seq_record.seq)
        
        #print(description)
        
        m = re.search('PE=\d', description)
        pe=int(m.group(0).split("=")[1])
        m = re.search('OS=.* (?=OX=)', description)
        organism=m.group(0).split("=")[1].strip()
        
        data_dict={"ID": accession, "name": name, "dataset":dataset, "proteinexistence":pe, "organism":organism, "sequence": sequence}
        rows.append(data_dict)
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)
    return df
    
def proteins_from_fasta(fasta_path):
    '''load proteins (as seqrecords) from fasta (just redirects)'''
    return seqrecords_from_fasta(fasta_path)

def seqrecords_from_fasta(fasta_path):
    '''load seqrecords from fasta file'''
    seqrecords = list(SeqIO.parse(fasta_path, "fasta"))
    return seqrecords

def seqrecords_to_sequences(seqrecords):
    '''converts biopythons seqrecords into a plain list of sequences'''
    return [str(p.seq) for p in seqrecords]

def sequences_to_fasta(sequences, fasta_path, sequence_id_prefix="s"):
    '''save plain list of sequences to fasta'''
    with open(fasta_path, "w") as output_handle:
        for i,s in tqdm(enumerate(sequences)):
            record = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(s), id=sequence_id_prefix+str(i), description="")
            SeqIO.write(record, output_handle, "fasta")

def df_to_fasta(df, fasta_path):
    '''Save column "sequence" from pandas DataFrame to fasta file using the index of the DataFrame as ID. Preserves original IDs in contrast to the function sequences_to_fasta()'''    
    with open(fasta_path, "w") as output_handle:
        for row in df.iterrows():            
            record = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(row[1]["sequence"]), id=str(row[0]), description="")
            SeqIO.write(record, output_handle, "fasta")

def sequences_to_df(sequences, sequence_id_prefix="s"):
    data = {'ID': [(sequence_id_prefix+str(i) if sequence_id_prefix!="" else i) for i in range(len(sequences))], 'sequence': sequences}
    df=pd.DataFrame.from_dict(data)
    return df.set_index("ID")

def fasta_to_df(fasta_path):
    seqs=SeqIO.parse(fasta_path, "fasta")
    res=[]
    for s in seqs:
        res.append({"ID":s.id,"sequence":str(s.seq)})
    return pd.DataFrame(res)

def peptides_from_proteins(protein_seqrecords, miss_cleavage=2,min_length=5,max_length=300):
    '''extract peptides from proteins seqrecords by trypsin digestion
    min_length: only return peptides of length min_length or greater (0 for all)
    max_length: only return peptides of length max_length or smaller (0 for all)
    '''
    peptides = []
    for seq in tqdm(protein_seqrecords):
        peps = trypsin_digest(str(seq.seq), miss_cleavage)
        peptides.extend(peps)
    tmp=list(set(peptides))
    if(min_length>0 and max_length>0):
        tmp=[t for t in tmp if (len(t)>=min_length and len(t)<=max_length)]
    elif(min_length==0 and max_length>0):
        tmp=[t for t in tmp if len(t)<=max_length]
    elif(min_length>0 and max_length==0):
        tmp=[t for t in tmp if len(t)>=min_length]
    print("Extracted",len(tmp),"unique peptides.")
    return tmp

def trypsin_digest(proseq, miss_cleavage):
    '''trypsin digestion of protein seqrecords
    TRYPSIN from https://github.com/yafeng/trypsin/blob/master/trypsin.py'''
    peptides=[]
    cut_sites=[0]
    for i in range(0,len(proseq)-1):
        if proseq[i]=='K' and proseq[i+1]!='P':
            cut_sites.append(i+1)
        elif proseq[i]=='R' and proseq[i+1]!='P':
            cut_sites.append(i+1)
    
    if cut_sites[-1]!=len(proseq):
        cut_sites.append(len(proseq))

    if len(cut_sites)>2:
        if  miss_cleavage==0:
            for j in range(0,len(cut_sites)-1):
                peptides.append(proseq[cut_sites[j]:cut_sites[j+1]])

        elif miss_cleavage==1:
            for j in range(0,len(cut_sites)-2):
                peptides.append(proseq[cut_sites[j]:cut_sites[j+1]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j+2]])
            
            peptides.append(proseq[cut_sites[-2]:cut_sites[-1]])

        elif miss_cleavage==2:
            for j in range(0,len(cut_sites)-3):
                peptides.append(proseq[cut_sites[j]:cut_sites[j+1]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j+2]])
                peptides.append(proseq[cut_sites[j]:cut_sites[j+3]])
            
            peptides.append(proseq[cut_sites[-3]:cut_sites[-2]])
            peptides.append(proseq[cut_sites[-3]:cut_sites[-1]])
            peptides.append(proseq[cut_sites[-2]:cut_sites[-1]])
    else: #there is no trypsin site in the protein sequence
        peptides.append(proseq)
    return list(set(peptides))
###########################################################################
# Processing CD-HIT clusters
###########################################################################
def clusters_df_from_sequence_df(df,threshold=[1.0,0.9,0.5],alignment_coverage=[0.0,0.9,0.8],memory=16000, threads=8, exploded=True, verbose=False):
    '''create clusters df from sequence df (using cd hit)
    df: dataframe with sequence information
    threshold: similarity threshold for clustering (pass a list for hierarchical clustering e.g [1.0, 0.9, 0.5])
    alignment_coverage: required minimum coverage of the longer sequence (to mimic uniref https://www.uniprot.org/help/uniref)
    memory: limit available memory
    threads: limit number of threads
    exploded: return exploded view of the dataframe (one row for every member vs. one row for every cluster)

    uses CD-HIT for clustering
    https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide
    copy cd-hit into ~/bin
    
    TODO: extend to psi-cd-hit for thresholds smaller than 0.4
    '''
    
    if verbose:
        print("Exporting original dataframe as fasta...")
    fasta_file = "cdhit.fasta"
    df_original_index = list(df.index) #reindex the dataframe since cdhit can only handle 19 letters
    df = df.reset_index(drop=True)

    df_to_fasta(df, fasta_file)

    if(not(isinstance(threshold, list))):
        threshold=[threshold]
        alignment_coverage=[alignment_coverage]
    assert(len(threshold)==len(alignment_coverage))    
    
    fasta_files=[]
    for i,thr in enumerate(threshold):
        if(thr< 0.4):#use psi-cd-hit here
            print("thresholds lower than 0.4 require psi-cd-hit.pl require psi-cd-hit.pl (building on BLAST) which is currently not supported")
            return pd.DataFrame()
        elif(thr<0.5):
            wl = 2
        elif(thr<0.6):
            wl = 3
        elif(thr<0.7):
            wl = 4
        else:
            wl = 5
        aL = alignment_coverage[i]

        #cd-hit -i nr -o nr80 -c 0.8 -n 5
        #cd-hit -i nr80 -o nr60 -c 0.6 -n 4
        #psi-cd-hit.pl -i nr60 -o nr30 -c 0.3
        if verbose:
            print("Clustering using cd-hit at threshold", thr, "using wordlength", wl, "and alignment coverage", aL, "...")

        fasta_file_new= "cdhit"+str(int(thr*100))+".fasta"
        command = "cd-hit -i "+fasta_file+" -o "+fasta_file_new+" -c "+str(thr)+" -n "+str(wl)+" -aL "+str(aL)+" -M "+str(memory)+" -T "+str(threads)
        if(verbose):
            print(command)
        process= subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()
        if(verbose):
            print(output)
        if(error !=""):
            print(error)
        fasta_files.append(fasta_file)
        if(i==len(threshold)-1):
            fasta_files.append(fasta_file_new)
        fasta_file= fasta_file_new
        
    #join results from all clustering steps
    if verbose:
        print("Joining results from different clustering steps...")
    for i,f in enumerate(reversed(fasta_files[1:])):
        if verbose:
            print("Processing",f,"...")
        if(i==0):
            df_clusters = parse_cdhit_clstr(f+".clstr",exploded=False)
        else:
            df_clusters2 = parse_cdhit_clstr(f+".clstr",exploded=False)
            for id,row in df_clusters.iterrows():
                members = row['members']
                new_members =  [list(df_clusters2[df_clusters2.repr_accession==y].members)[0] for y in members]
                new_members = [item for sublist in new_members for item in sublist] #flattened
                row['members']=new_members

    df_clusters["members"]=df_clusters["members"].apply(lambda x:[df_original_index[int(y)] for y in x])
    df_clusters["repr_accession"]=df_clusters["repr_accession"].apply(lambda x:df_original_index[int(x)])

    if(exploded):
        return explode_clusters_df(df_clusters)
    return df_clusters

def parse_cdhit_clstr(filename, exploded=True):
    '''Aux. Function (used by clusters_df_from_sequence_df) to parse CD-HITs clstr output file in a similar way as the uniref data
    for the format see https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#CDHIT

    exploded:  single row for every ID instead of single for every cluster   
        '''
    def save_cluster(rows,members,representative):
        if(len(members)>0):
            rows.append({"entry_id":filename[:-6]+"_"+representative, "members":members, "repr_accession":representative})
            
    rows=[]
    with open(filename, 'r') as f:
        members=[]
        representative=""

        for l in tqdm(f):
            if(l[0]==">"):
                save_cluster(rows,members,representative)
                members=[]
                representative=""
            else:
                member=(l.split(">")[1]).split("...")[0]
                members.append(member)
                if "*" in l:
                    representative = member
    save_cluster(rows,members,representative)
    df=pd.DataFrame(rows).set_index("entry_id")
    
    if(exploded):
        return explode_clusters_df(df)
    return df
###########################################################################
# MHC DATA
###########################################################################

######### Helper functions ##########
def _label_binder(data, threshold=500, measurement_column="meas"): 
    # Drop entries above IC50 > 500nM with inequality < (ambiguous)
    to_drop = (( (data['inequality']=='<')&(data[measurement_column]>threshold))|((data['inequality']=='>')&(data[measurement_column]<threshold))).mean()
    if to_drop > 0:
        print('Dropping {} % because of ambiguous inequality'.format(to_drop))
        data = data[~(( (data['inequality']=='<')&(data[measurement_column]>threshold))|((data['inequality']=='>')&(data[measurement_column]<threshold)))]

    # Labeling
    data['label'] = (1* data[measurement_column]<=threshold).astype(int)

    return data
 
def _transform_ic50(data, how="log",max_ic50=50000.0, inequality_offset=True, label_column="meas"):
    """Transform ic50 measurements 
    
    how: "log" logarithmic transform, inequality "=" mapped to [0,1], inequality ">" mapped to [2,3], inequality "<" mapped to [4,5]
         "norm"
         "cap" 
    """
    x = data[label_column]
    
    if how=="cap":
        x = np.minimum(x, 50000)
    elif how=="norm":
        x = np.minimum(x, 50000)
        x = (x - x.mean()) / x.std()
    elif how=="log":
        # log transform 
        x = 1 - (np.log(x)/np.log(max_ic50))
        x = np.minimum(1.0, np.maximum(0.0,x))
        
        if(inequality_offset):
            # add offsets for loss
            offsets = pd.Series(data['inequality']).map({'=': 0, '>': 2, '<': 4,}).values
            x += offsets
        
    return x
 
def _string_index(data):
    # Add prefix letter "a" to the numerical index (such that it is clearly a string in order to avoid later errors).
    data["ID"] = data.index
    data["ID"] = data["ID"].apply(lambda x: "a"+ str(x))
    data = data.set_index(["ID"])
    
    return data
    
def _format_alleles(x):
    if x[:3]=='HLA':
        return x[:5]+'-'+x[6:8]+x[9:]
    if x[:4]=='Mamu':
        return x[:6]+'-'+x[7:]
    else:
        return x

def _get_allele_ranking(data_dir='.'):
    '''
    Allele ranking should be the same across different datasets (noMS, withMS) to avoid confusion. 
    Thus, the ranking is based on the larger withMS dataset 
    '''
    data_dir = Path(data_dir)
    curated_withMS_path = data_path/'data_curated.20180219'/'curated_training_data.with_mass_spec.csv'
    df = pd.read_csv(curated_withMS_path)
    
     # Drop duplicates
    df = df.drop_duplicates(["allele", "peptide","measurement_value"])

    lens = df['peptide'].apply(len)
    df = df[(lens>7) & (lens<16)]
    
    # Keep only alleles with min 25 peptides like MHC flurry
    peptides_per_allele = df.groupby('allele').size()
    alleles_select = peptides_per_allele[peptides_per_allele>24].index
    df = df[df['allele'].isin(alleles_select)]
    
    mhc_rank = df.groupby('allele').size().sort_values(ascending=False).reset_index()['allele']
    
    return mhc_rank
    
########## Generate DataFrame ##########
def generate_mhc_kim(cv_type=None, mhc_select=0, regression=False,  transform_ic50=None, to_csv=False, filename=None, data_dir='.', keep_all_alleles=False):
    '''
    cv_type: string, strategy for 5-fold cross validation, options:
        - None: No cv-strategy, cv column is filled with 'TBD'
        - sr: removal of similar peptides seperatly in binder/ non-binder set, using similarity threshold of 80%, similarity found with 'Hobohm 1 like algorithm'
        - gs: grouping similar peptides in the same cv-partition
        - rnd: random partioning
    transform_ic50: string, ignnored if not regression
        - None: use raw ic50 measurements as labels
        - cap: cap ic50 meas at 50000
        - norm: cap ic50 meas at 50000 and normalize
        - log: take log_50000 and cap at 50000
    mhc_select: int between 0 and 50, choose allele by frequency rank in Binding Data 2009
    
    '''
    # Binding Data 2009. Used by Kim et al for Cross Validation. Used by MHCnugget for training.
    bd09_file = 'bdata.2009.mhci.public.1.txt'
    # Similar peptides removed
    bd09_cv_sr_file = 'bdata.2009.mhci.public.1.cv_sr.txt'
    # Random partioning
    bd09_cv_rnd_file = 'bdata.2009.mhci.public.1.cv_rnd.txt'
    # Similar peptides grouped
    bd09_cv_gs_file = 'bdata.2009.mhci.public.1.cv_gs.txt'
    
    # 'blind' used by Kim et al to estimate true predicitve accuracy. Used by MHCnugget for testing.
    # Generated by subtracting BD2009 from BD 2013 and removing similar peptides with respect to BD2009 
    # (similar = at least 80% similarity and same length)
    bdblind_file = 'bdata.2013.mhci.public.blind.1.txt'
    
    data_dir = Path(data_dir)/"benchmark_mhci_reliability/binding"
    
    # Read in data with specified cv type
    if cv_type=='sr':
        bd09 = pd.read_csv(data_dir/'bd2009.1'/bd09_cv_sr_file, sep='\t')
    elif cv_type=='gs':
        bd09 = pd.read_csv(data_dir/'bd2009.1'/bd09_cv_gs_file, sep='\t')
    elif cv_type=='rnd':
        bd09 = pd.read_csv(data_dir/'bd2009.1'/bd09_cv_rnd_file, sep='\t')
    else:
        bd09 = pd.read_csv(data_dir/'bd2009.1'/bd09_file, sep='\t')
    # Read in blind data
    bdblind = pd.read_csv(data_dir/'blind.1'/bdblind_file, sep='\t')
    # alleles are spelled differently in bdblind and bd2009, change spelling in bdblind
    bdblind['mhc'] = bdblind['mhc'].apply(_format_alleles)
    
    # Confirm there is no overlap
    print('{} entries from the blind data set are in the 2009 data set'.format(bdblind[['sequence', 'mhc']].isin(bd09[['sequence', 'mhc']]).all(axis=1).sum()))
    
    if regression:
        # For now: use only quantitative measurements, later tuple (label, inequality as int)
        #print('Using quantitative {} % percent of the data.'.format((bd09['inequality']=='=').mean()))
        #bd09 = bd09[bd09['inequality']=='=']
        #bd09.rename(columns={'meas':'label'}, inplace=True)
        #bdblind = bdblind[bdblind['inequality']=='=']
        #bdblind.rename(columns={'meas':'label'}, inplace=True)
        # Convert ic50 measurements to range [0,1]
        if transform_ic50 is not None:
            bd09['label'] = _transform_ic50(bd09, how=transform_ic50)
            bdblind['label'] = _transform_ic50(bdblind, how=transform_ic50)
    else:
        # Labeling for binder/NonBinder
        bd09 = _label_binder(bd09)[['mhc', 'sequence', 'label', 'cv']]
        #bdblind = _label_binder(bdblind)[['mhc', 'sequence', 'label', 'cv']]
        bdblind = bdblind.rename(columns={"meas":"label"})
    
    if not keep_all_alleles:
        # in bd09 (train set) keep only entries with mhc also occuring in bdblind (test set)
        bd09 = bd09[bd09['mhc'].isin(bdblind['mhc'])]
    
    # Combine
    bdblind['cv'] = 'blind'
    bd = pd.concat([bd09, bdblind], ignore_index=True)
    
    if not(regression):
        # Test if there is at least one binder in bd09 AND bdblind
        min_one_binder = pd.concat([(bd09.groupby('mhc')['label'].sum() > 0), (bdblind.groupby('mhc')['label'].sum() > 0)], axis=1).all(axis=1)
        print('For {} alleles there is not at least one binder in bd 2009 AND bd blind. These will be dismissed.'.format((~min_one_binder).sum()))
        alleles = bd['mhc'].unique()
        allesles_to_keep = alleles[min_one_binder]
        # Dismiss alleles without at least one binder
        bd = bd[bd['mhc'].isin(allesles_to_keep)]
        
    # Make allele ranking based on binding data 2009
    mhc_rank = bd[bd['cv']!='blind'].groupby('mhc').size().sort_values(ascending=False).reset_index()['mhc']
    
    # Select allele
    if mhc_select is not None:
        print('Selecting allele {}'.format(mhc_rank.loc[mhc_select]))
        bd = bd[bd['mhc']==mhc_rank.loc[mhc_select]][['sequence', 'label', 'cv']]
    
    # Turn indices into strings
    bd = _string_index(bd)
    
    if to_csv and filename is not None:
        bd.to_csv(filename)
    
    return bd
    
def generate_mhc_flurry(ms='noMS', mhc_select=0, regression=False, transform_ic50=None, binder_threshold=500, filter_length=True, label_binary=False, random_seed=42,data_dir='.'):
    '''
    Load the MHC I data curated and uploaded to https://data.mendeley.com/datasets/8pz43nvvxh/1 by MHCFlurry
    Used by them for training and model selection
    
    ms: string, specifies if mass spectroscopy data should be included, options:
     - noMS: MHCFlurry no MS dataset
     - withMS: MHCFlurry with MS dataset
    mhc_select: int between 0 and  150 (noMS)/ 188 (withMS), choose allele by frequency rank
    filter_length: boolean, MHCFlurry selected peptides of length 8-15 (their model only deals with these lengths)

    '''
    
    data_path = Path(data_dir)
    curated_noMS_path = data_path/'data_curated.20180219'/'curated_training_data.no_mass_spec.csv'
    curated_withMS_path = data_path/'data_curated.20180219'/'curated_training_data.with_mass_spec.csv'
    
    if ms=='noMS':
        df = pd.read_csv(curated_noMS_path)
    elif ms=='withMS':
        df = pd.read_csv(curated_withMS_path)
    
    if filter_length:
        lens = df['peptide'].apply(len)
        df = df[(lens>7) & (lens<16)]
    
    # Keep only alleles with min 25 peptides
    peptides_per_allele = df.groupby('allele').size()
    alleles_select = peptides_per_allele[peptides_per_allele>24].index
    df = df[df['allele'].isin(alleles_select)]
    
    df.rename(columns={'measurement_value':'meas', 'measurement_inequality':'inequality', 'peptide':'sequence'}, inplace=True)
    
    # label binder/non binder
    if label_binary:
        df = _label_binder(df, threshold=binder_threshold, measurement_column='label')
    
    if regression:
        df["label"] = _transform_ic50(df, how=transform_ic50)
    
    if mhc_select is not None:
        if type(mhc_select)==int: 
            mhc_rank = df.groupby('allele').size().sort_values(ascending=False).reset_index()['allele']
            print('Selecting allele {}'.format(mhc_rank.loc[mhc_select]))
            df = df[df['allele']==mhc_rank.loc[mhc_select]]
        else:
            print('Selecting allele {}'.format(mhc_select))
            df = df[df['allele']==mhc_select]
    
    # Mark 10% of the data as validation set
    np.random.seed(seed=random_seed)
    val_ind = np.random.randint(0,high=df.shape[0],size=int(df.shape[0]/10))
    df['cluster_ID'] = (df.reset_index().index.isin(val_ind))*1
    df["ID"]=df.sequence.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
     
    #df = _string_index(df)
    
    return df

def generate_abelin(mhc_select=0, data_dir='.'):
    '''
    mhc_select: int in [1, 2, 4, 6, 8, 10, 13, 14, 15, 16, 17, 21, 22, 36, 50, 63]
    '''  
    data_path = Path(data_dir)
    abelin = pd.read_csv(data_path/"abelin_peptides.all_predictions.csv")[['hit', 'allele', 'peptide']]
    abelin.rename(columns={'peptide':'sequence'}, inplace=True)
    
    # Remove entries present in training set (training data here: noMS as only MHCFlurry noMS is benchmarked with Abelin data)
    train = generate_mhc_flurry(ms='noMS',mhc_select=None, data_dir=data_dir)[['allele', 'sequence']]
    overlap_ind = abelin[['allele', 'sequence']].merge(train.drop_duplicates(['allele','sequence']).assign(vec=True),how='left', on=['allele', 'sequence']).fillna(False)['vec']
    #print(abelin.shape[0], overlap_ind.shape, overlap_ind.sum() )
    abelin = abelin[~overlap_ind.values]
    
    # Select allele specific data
    if type(mhc_select)==int: 
        allele_ranking = _get_allele_ranking(data_dir=data_dir)
        mhc_select = allele_ranking.iloc[mhc_select]        
    abelin = abelin[abelin['allele']==mhc_select]
    
    abelin.rename(columns={'hit':'label'}, inplace=True)
    
    abelin['cluster_ID'] = 2
    
    return abelin
 
def prepare_hpv(mhc_select, data_dir='.'): 
        '''
        To run, download Table S2 from Supplementary Material of [Bonsack M, Hoppe S, Winter J, Tichy D, Zeller C, Küpper MD, et al. Performance evaluation of MHC class-I binding prediction tools based on an experimentally validated MHC–peptide binding data set. Cancer Immunol Res 2019;7:719–36.] and save as HPV_data.csv in ./data
        
        mhc_select: string from ['HLAA1', 'HLAA2', 'HLAA3', 'HLAA11', 'HLAA24', 'HLAB7', 'HLAB15']
        '''
        data_path = Path(data_dir)
        df = pd.read_csv(data_path/"HPV_data.csv")
        
        df["label"] = df["Experimental binding capacity"].mask(df["Experimental binding capacity"]=="nb")
        
        return df[df["allele"]==mhc_select][["sequence","label"]]
        
        
def prepare_mhcii_iedb2016(mhc_select, cv_fold, path_iedb="../data/iedb2016", path_jensen_csv="../data/jensen_et_al_2018_immunology_supplTabl3.csv"):
    '''prepares mhcii iedb 2016 dataset using train1 ... test5 from http://www.cbs.dtu.dk/suppl/immunology/NetMHCIIpan-3.2/'''
    def prepare_df(filename):
        df = pd.read_csv(filename,header=None,sep="\t")
        df.columns=["sequence","aff_log50k","allele"]
        df["ID"]=df.sequence.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
        df["aff"]=df.aff_log50k.apply(lambda x:np.power(50000,1-x))
        return df
    path_iedb = Path(path_iedb)

    dfs_traintest = []
    for i in range(1,6):
        dfs_traintest.append(prepare_df(path_iedb/("test"+str(i))))
        dfs_traintest[-1]["cv_fold"]=i-1
        dfs_traintest[-1]["traintest"]="test"
        dfs_traintest.append(prepare_df(path_iedb/("train"+str(i))))
        dfs_traintest[-1]["cv_fold"]=i-1
        dfs_traintest[-1]["traintest"]="train"
    df_traintest = pd.concat(dfs_traintest,ignore_index=True)
    
    df_traintest.rename(columns={"aff_log50k":"label"},inplace=True)
    
    # select only alleles with results in the Jensen et al paper
    df_pub = pd.read_csv(path_jensen_csv)
    df_traintest = df_traintest[df_traintest["allele"].isin(df_pub["Molecule"].unique())]
   
    # select one allele based on training set size ranking
    allele_ranking = df_traintest[df_traintest["cv_fold"]==0].groupby("allele").size().sort_values(ascending=False).index
    df_traintest = df_traintest[df_traintest["allele"]==allele_ranking[mhc_select]]
    
    #select specified cv_fold
    df_traintest = df_traintest[df_traintest["cv_fold"]==cv_fold]
    
    #stratified split of train -> train & val
    binder = 1.0*(df_traintest[df_traintest["traintest"]=="train"]["aff"] < 500).values
    tts = train_test_split(df_traintest[df_traintest["traintest"]=="train"], test_size=0.1, random_state=42, stratify=binder)
    df_train = tts[0]
    df_train["cluster_ID"] = 0
    df_val = tts[1]
    df_val["cluster_ID"] = 1
    df_test = df_traintest[df_traintest["traintest"]=="test"]
    df_test["cluster_ID"] = 2
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    return df

def prepare_mhcii_iedb2009(path_2009= "../data/classII_binding_data_Nov_16_2009",similarity_reduced=True):
    '''prepares mhcii iedb 2009 dataset using classII_binding_data_Nov_16_2009.zip from http://tools.iedb.org/mhcii/download/'''
    def mhc_ii_allelename_format(x):
        if x.startswith("HLA"):
            if len(x)==23:
                return x[:8]+x[9:13]+"-"+x[14:18]+x[19:]
            elif len(x)==13:
                return x[:8]+x[9:]
            else:
                return x
        else:
            return x
        
    path_2009 = Path(path_2009)
    if(similarity_reduced):
        path_2009 = path_2009/"class_II_similarity_reduced_5cv_sep"
    else:
        path_2009= path_2009/ "class_II_all_split_5cv"

    dfs_2009=[]
    for f in Path(path_2009).glob("*.txt"):
        df_tmp = pd.read_csv(f,header=None,sep="\t").drop([3,5],axis=1)
        df_tmp.columns=["species","allele","sequence_len","sequence","aff"]
        df_tmp["cv_fold"] = f.stem.split("_")[-1]
        filename = np.array(f.stem.split("_"))
        df_tmp["traintest"] = filename[np.logical_or(filename=="train", filename=="test")][0] #f.stem.split("_")[1]
        dfs_2009.append(df_tmp)

    df_2009=pd.concat(dfs_2009)
    df_2009.rename(columns={"aff":"label"}, inplace=True)
    df_2009["ID"]=df_2009.sequence.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    
    # rename the alleles to match the format of the corresponding test dataset from https://doi.org/10.1371/journal.pcbi.1006457.s010
    df_2009["allele"] = df_2009["allele"].apply(mhc_ii_allelename_format)
    
    # transform ic50 values
    df_2009["label"] = _transform_ic50(df_2009,how="log",inequality_offset=False,label_column="label")
    return df_2009

def prepare_mhci_pcbi(path_pcbi="../data/journal.pcbi.1006457.s009",mer=None):
    '''prepares mhci test dataset from https://doi.org/10.1371/journal.pcbi.1006457.s009'''
    if mer is None:
        path_pcbi = [Path(path_pcbi)/"9mer",Path(path_pcbi)/"10mer"]
    else:
        path_pcbi=[Path(path_pcbi)/"{}mer".format(mer)]
    #print(path_pcbi)
    dfs_pcbi=[]
    for p in path_pcbi:
        for f in Path(p).glob("*.txt"):
            df_tmp = pd.read_csv(f,header=None,sep="\t")
            if(len(df_tmp.columns)!=2):
                #print("Warning:",f,"does not have the correct format. Skipping.")
                continue
            df_tmp.columns=["sequence","aff"]
            df_tmp["dataset"] = f.stem
            dfs_pcbi.append(df_tmp)
    
    df_pcbi=pd.concat(dfs_pcbi,ignore_index=True)
    df_pcbi.rename(columns={"dataset":"allele","aff":"label"}, inplace=True)

    df_pcbi["ID"]=df_pcbi.sequence.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    
    def kim_allele_format(x):
        return x[:5] + "-" + x[8:10] + x[13:]
    df_pcbi["allele"] = df_pcbi["allele"].apply(kim_allele_format) 

    # transform ic50 values
    df_pcbi["label"] = _transform_ic50(df_pcbi,how="log",inequality_offset=False,label_column="label")
    return df_pcbi
    
def prepare_mhcii_pcbi(path_pcbi="../data/journal.pcbi.1006457.s010"):
    '''prepares mhcii test dataset from https://doi.org/10.1371/journal.pcbi.1006457.s010'''
    path_pcbi=Path(path_pcbi)/"15mer"

    dfs_pcbi=[]
    for f in Path(path_pcbi).glob("*.txt"):
        df_tmp = pd.read_csv(f,header=None,sep="\t")
        
        if(len(df_tmp.columns)!=2):
            #print("Warning:",f,"does not have the correct format. Skipping.")
            continue

        df_tmp.columns=["sequence","aff"]
        df_tmp["dataset"] = f.stem
        dfs_pcbi.append(df_tmp)

    df_pcbi=pd.concat(dfs_pcbi)
    df_pcbi.rename(columns={"dataset":"allele","aff":"label"}, inplace=True)
    
    # select only the alleles present in the mhcii iedb 2009 training dataset
    # df_2009 = prepare_mhcii_iedb2009(path_2009= path_2009,similarity_reduced=similarity_reduced)
    # df_pcbi = df_pcbi[df_pcbi["allele"].isin(df_2009["allele"].unique())]
    
    df_pcbi["ID"]=df_pcbi.sequence.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    
    # transform ic50 values
    df_pcbi["label"] = _transform_ic50(df_pcbi,how="log",inequality_offset=False,label_column="label")
    return df_pcbi
    
#############################cleavage stuff    
#parsing cleavage positions from chipper/netchop
def parse_cleavage_chipper(filename, probs=True, return_sequences=True):
    '''
    probs: parse chipper -p output (probabilities)
    returns list of cleavage positions for each protein
    parses chipper output e.g. result of ./chipper -i ../share/chipper/test.fa | gunzip >test2.dat'''
    with open(filename, 'r') as f:
        tmp=f.readlines()

    indices=[]

    for i,t in enumerate(tmp):
        if("@" in t):
            indices.append(i)

    output=[]
    output_seq = []
    for i in indices:
        name = tmp[i]
        cleavage_string = tmp[i+3].strip()
        if(probs is False):
            cleavage_ids = np.where(np.array([t for t in cleavage_string])=='+')[0]
        else:
            cleavage_ids = [np.where(np.array([t for t in cleavage_string])==str(i))[0] for i in range(10)]
        output.append(cleavage_ids)
        #print("\n\nname",name)
        #print(cleavage_string)
        #print(len(cleavage_string))
        #print(len(cleavage_ids))
        if(return_sequences is True):
            output_seq.append(tmp[i+1].strip())
    if(return_sequences is True):
        return zip(output_seq, output)
    else:
        return output

def parse_cleavage_netchop_short(filename, return_sequences=True, startidx=4):
    '''
    returns cleavage positions
    parses chipper output e.g. result of e.g. result of ./chipper -s -i ../share/chipper/test.fa >test2.dat
    note: the chipper output is buggy and does not return the last AA (but it cannot make predictions for it anyway)
    startidx=4 for chipper and startidx=21 for netchop
    '''
    with open(filename, 'r') as f:
        tmp=f.readlines()

    indices=[]

    for i,t in enumerate(tmp):
        if("-----" in t):
            indices.append(i-1)

    endindices=indices[::2]
    startindices=[startidx]+[t+4 for t in indices[1::2][:-1]]
    
    output = []
    output_seq = []
    for s,e in zip(startindices,endindices):
        #print(s,e)
        ids= range(s,e+1,2)
        ids_seq = range(s-1,e,2)
        #print(list(ids))
        name = tmp[s-2]
        cleavage_string = "".join([tmp[i].strip() for i in ids])
        if(return_sequences):
            output_seq.append("".join([tmp[i].strip() for i in ids_seq]))
        cleavage_ids = np.where(np.array([t for t in cleavage_string])=='S')[0]
        output.append(cleavage_ids)
        #print("\n\nname",name)
        #print(cleavage_string)
        #print(len(cleavage_string))
        #print(len(cleavage_ids))
    if(return_sequences is True):
        return zip(output_seq, output)
    else:
        return output

def cut_seq_chipper(seq,cleavs,threshold=5,min_length=5,max_length=20, skip_cleav=0):
    '''cuts sequence according to cleavage predictions
    seq: sequence as string
    cleavs: cleavage predictions as np.array (e.g. output of parse_cleavage_...)
    threshold: cutting threshold 0...9 (e.g. 8 corresponds to threshold 0.8)- None in case cleavage file was parsed with probs=False
    min_length: minimum length of resulting peptides
    max_length: maximum length of resulting peptides
    skip_cleav: number of cleavage predictions to skip
    '''
    if(threshold is not None):
        cleavs = np.concatenate(cleavs[threshold:],axis=0)
    cleavs=np.sort(cleavs)
    peptides =[]
    
    for i in range(len(cleavs)):
        #cuts after the specified position c.f. netchop documentation
        for j in range(skip_cleav+1):
            if (i+j)<len(cleavs):
                peptides.append(seq[0 if i==0 else cleavs[i-1] + 1:  cleavs[i+j]+1])
    if(len(cleavs)>0):
        for j in range(skip_cleav+1):
            if j<len(cleavs):
                peptides.append(seq[cleavs[-1-j]+1:])
    peptides = [p for p in peptides if (len(p)>=min_length and (max_length==0 or len(p)<=max_length))]
    return  np.unique(peptides)

def cut_chipper(filename="test.dat",threshold=5, min_length=5, max_length=20):
    '''
    loads chipper output and returns corresponding set of peptides
    usage: run chipper on fasta file ./chipper -i input.fasta | gunzip >cleavage.dat
    pass cleavage.dat to cut_chipper
    '''
    out=parse_cleavage_chipper(filename, probs=True, return_sequences=True)
    
    peptides=[]
    for seq,cleavs in out:
        peptides.append(cut_seq_chipper(seq,cleavs,threshold=threshold,min_length=min_length,max_length=max_length))
    peptides = np.unique(np.concatenate(peptides,axis=0))
    return peptides

def cut_netchop(filename="test.dat", min_length=5, max_length=20, skip_cleav=0):
    '''
    loads netchop short output and returns corresponding set of peptides
    usage: run chipper on fasta file ./chipper -i input.fasta | gunzip >cleavage.dat
    pass cleavage.dat to cut_chipper
    '''
    out=parse_cleavage_netchop_short(filename, return_sequences=True, startidx=21)
    peptides=[]
    for seq,cleavs in out:
        peptides.append(cut_seq_chipper(seq,cleavs,threshold=None,min_length=min_length,max_length=max_length,skip_cleav=skip_cleav))
    peptides = np.unique(np.concatenate(peptides,axis=0))
    return peptides

#sequence cleaner (drops sequences with unknown AAs from fasta)
#adopted from https://biopython.org/wiki/Sequence_Cleaner
def sequence_cleaner(fasta_file_in, fasta_file_out, min_length=0, drop_aas=["X","Z","B"]):
    # Create our hash table to add the sequences
    sequences={}

    # Using the Biopython fasta parse we can read our fasta input
    for seq_record in SeqIO.parse(str(fasta_file_in), "fasta"):
        # Take the current sequence
        sequence = str(seq_record.seq).upper()
        
        # Check if the current sequence is according to the user parameters
        if(len(sequence) >= min_length and np.all([sequence.count(a)==0 for a in drop_aas])):
        # If the sequence passed in the test "is it clean?" and it isn't in the
        # hash table, the sequence and its id are going to be in the hash
            if sequence not in sequences:
                sequences[sequence] = seq_record.id
       # If it is already in the hash table, we're just gonna concatenate the ID
       # of the current sequence to another one that is already in the hash table
            else:
                sequences[sequence] += "_" + seq_record.id


    # Write the clean sequences
    
    # Create a file in the same directory where you ran this script
    with open(str(fasta_file_out), "w+") as output_file:
        # Just read the hash table and write on the file as a fasta format
        for sequence in sequences:
            if "|" in sequences[sequence]:
                ID = sequences[sequence].split('|')[1]
            else:
                ID = sequences[sequence]
            output_file.write(">" + ID + "\n" + sequence + "\n")

    print("Clean fasta file written to " + str(fasta_file_out))

def chipper_digest(fasta_file, chipper_path="chipper", threshold=7, min_length=5, max_length=20,verbose=True):
    '''Cuts proteins from given fasta_file into peptides using chipper cleavage predictions
    call with chipper_path e.g. ./chipper for a custom chipper path (otherwise chipper has to be in searchpath)
    '''
    # 1. save clean fasta file
    if(verbose):
        print("Saving clean fasta file...")
    fasta_file_clean = fasta_file.parent/("clean_"+fasta_file.stem+".fasta")
    sequence_cleaner(fasta_file, fasta_file_clean , min_length=min_length, drop_aas=["X","Z","B","U","O"])#also drop U and O as chipper cannot handle them
    # 2. run chipper (exec has to be in searchpath e.g. ~/bin)
    if(verbose):
        print("Running chipper for cleavage prediction...")
    command = chipper_path+" -p -i "+str(fasta_file_clean)+" -o cleavage.dat.gz"
    if(verbose):
        print(command)
    process= subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process.communicate()
    if(verbose):
        print(output)
    if(error !=""):
        print(error)
    command2 = "gunzip -f cleavage.dat.gz"
    if(verbose):
        print(command2)
    process2= subprocess.Popen(command2.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, error = process2.communicate()
    if(verbose):
        print(output)
    if(error !=""):
        print(error)
    # 3. cut proteins according to predictions
    if(verbose):
        print("Cutting proteins according to predictions...")
    peptides = cut_chipper("cleavage.dat",threshold=threshold,min_length=min_length,max_length=max_length)
    print("extracted ",len(peptides), "peptides.")
    fasta_file_clean.unlink()
    #Path("cleavage.dat.tar.gz").unlink()
    Path("cleavage.dat").unlink()
    return peptides

def netchop_digest(fasta_file, netchop_path="netchop", threshold=0.7, min_length=5, max_length=20, repeats=10, verbose=True):
    '''Cuts proteins from given fasta_file into peptides using netchop cleavage predictions
    netchop_path: e.g. ./netchop to call netchop in present directory (the default value requires to place symlink to netchop tcsh in netchop3.1 into searchpath e.g. ~/bin) BE CAREFUL NETCHOP TMP PATH MAY NOT BE TOO LONG
    '''
    # 1. save clean fasta file
    if(verbose):
        print("Saving clean fasta file...")
    if(isinstance(fasta_file,str)):
        fasta_file = Path(fasta_file)
    fasta_file_clean = fasta_file.parent/("clean_"+fasta_file.stem+".fasta")
    sequence_cleaner(fasta_file, fasta_file_clean , min_length=min_length, drop_aas=["X","Z","B","U","O"])#also drop U and O as netChop cannot handle them?

    # 2. run netChop (exec has to be in searchpath e.g. ~/bin) BE CAREFUL NETCHOP TMP PATH MAY NOT BE TOO LONG
    fasta_file_clean_output = fasta_file_clean.parent / (fasta_file_clean.stem+".out")
    if(verbose):
        print("Running netchop for cleavage prediction...")
    command = netchop_path+" "+str(fasta_file_clean)+" -t "+str(threshold)
    if(verbose):
        print(command)
    with open(fasta_file_clean_output,"wb") as out:
        process= subprocess.Popen(command.split(), stdout=out, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()
        #if(verbose):
        #    print(output)
        if(error !=""):
            print(error)
    cleavage_df = parse_netchop_long(fasta_file_clean_output)
    
    # 3. cut proteins according to predictions
    if(verbose):
        print("Cutting proteins according to predictions...")
    peptides = cut_netchop_stochastic(cleavage_df,repeats=repeats,min_len=min_length,max_len=max_length)
    print("extracted ",len(peptides), "peptides.")
    #fasta_file_clean.unlink()
    #fasta_file_clean_output.unlink()
    return peptides

def parse_netchop_long(filename):
    '''parses netchops output (long form) into dataframe'''
    with open(filename, "r") as f:
        tmp=f.readlines()
    start=np.where([x == ' pos  AA  C      score      Ident\n' for x in tmp])[0]+2
    endx=np.where([x == '\n' for x in tmp])[0]-1
    end=[]
    for s in start:
        for e in endx:
            if(e>s):
                end.append(e)
                break
    cleavages =[]
    accessions = []
    sequences=[]
    #print(start,end)
    for s,e in zip(start,end):
        #print(s,e,tmp[s:e])
        raw = [[t.strip() for t in x.split(" ") if t!=""] for x in tmp[s:e]]
        cleavages.append([float(r[3]) for r in raw])
        accessions.append(raw[0][-1].split(" ")[-1].replace("tr|","").replace("sp|","").replace("|",""))
        sequences.append("".join([r[1] for r in raw]))
    
    df=pd.DataFrame({"accession":accessions,"cleavages":cleavages, "sequence":sequences})
    return df.set_index("accession")

def netchop_cleavage_from_fasta(fasta_file,netchop_path="netchop",min_length=5,verbose=True):
    '''return cleavage probs for given fasta file'''
    if(isinstance(fasta_file,str)):
        fasta_file = Path(fasta_file)
    # 2. run netChop (exec has to be in searchpath e.g. ~/bin) BE CAREFUL NETCHOP TMP PATH MAY NOT BE TOO LONG
    fasta_file_clean_output = fasta_file.parent / (fasta_file.stem+".out")
    if(verbose):
        print("Running netchop for cleavage prediction...")
    command = netchop_path+" "+str(fasta_file)
    if(verbose):
        print(command)
    with open(fasta_file_clean_output,"wb") as out:
        process= subprocess.Popen(command.split(), stdout=out, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()
        #if(verbose):
        #    print(output)
        if(error !=""):
            print(error)
            
    df = parse_netchop_long(fasta_file_clean_output)
    fasta_file_clean_output.unlink()
    return df

def netchop_cleavage_from_df(df,netchop_path="netchop",bs=100,min_length=5):
    '''obtain cleavage probs for a given df with sequences (processed batch-wise)'''
    #remove sequences with nonstandard amino acids
    df_clean = df[df.sequence.apply(lambda x:np.all([not(s in x) for s in ["X","Z","B","U","O"]]))==True].copy()
    start_idx=range(0,len(df_clean),bs)
    
    dfs_cleavage=[]
    for s in tqdm(start_idx):
        df_to_fasta(df_clean.iloc[s:min(s+bs,len(df_clean))],"tmp.fasta")
        dfs_cleavage.append(netchop_cleavage_from_fasta(Path("tmp.fasta"),netchop_path=netchop_path,min_length=min_length,verbose=False))
    Path("tmp.fasta").unlink()
    return pd.concat(dfs_cleavage)

def sample_using_cleavages(seq,cleavages,repeats=10,min_len=5,max_len=20):
    '''cuts a single sequence stochastically using the given cleavage probabilities'''
    #cuts after the specified position c.f. netchop documentation
    seq_len = len(cleavages)
    fragments = []
    for _ in range(repeats):
        cuts= (np.random.uniform(size=seq_len)<cleavages)
        cuts_ends = list(np.where(cuts)[0] + 1)
        if(len(cuts_ends)==0 or cuts_ends[-1] != seq_len):
            cuts_ends.append(seq_len)
        for i in range(len(cuts_ends)):
            cut_start = 0 if i==0 else cuts_ends[i-1]
            cut_end = cuts_ends[i]
            fragment_len = cut_end-cut_start
            if(fragment_len>=min_len and fragment_len<=max_len):
                fragments.append(seq[cut_start:cut_end])
    return fragments

def cut_netchop_stochastic(cleavage_df,repeats=10,min_len=5,max_len=20):
    '''cuts proteins stochastically  based on netchops cleavage probabilities (passed via cleavage df)'''
    fragments=[]
    for i,row in tqdm(cleavage_df.iterrows()):
        tmp = sample_using_cleavages(row.sequence,row.cleavages,repeats=repeats,min_len=min_len,max_len=max_len)
        fragments.append(tmp)
    return  [item for sublist in fragments for item in sublist]
    return pd.concat(dfs_cleavage)

def hobohl_similarity(query, seqs,threshold=0.8):
    '''returns all sequences that are similar in the hobohl sense i.e. have same length and sequence identity>threshold'''
    lst=[]
    seqs_same_length = np.array([len(x)==len(query) for x in seqs])
    for x in seqs[np.where(seqs_same_length)[0]]:
        if(x!=query and sum([int(i==j) for i,j in zip(query,x)])/len(query)>=threshold):
            lst.append(x)
    return lst

def compute_hobohl_clusters(seqs,threshold=0.8,exploded=True):
    '''computes clusters of hobohl similar peptides from list of peptides'''
    
    neighbors = np.array([hobohl_similarity(q,seqs,threshold=threshold) for q in seqs])
    neighbors_length = np.array([len(x) for x in neighbors])
    idxs = np.argsort(neighbors_length)
    neighbors_sorted = list(neighbors[idxs].copy())
    seqs_sorted = list(seqs[idxs].copy())
    clusters = []
    
    while(len(seqs_sorted)>0):
        seq = seqs_sorted[0]
        neigh = neighbors_sorted[0]
        existing=np.where([np.any([n in c for n in neigh]) for c in clusters])[0]
        if(len(existing)>0):
            if(len(existing)>1):
                #join existing clusters
                for e2 in reversed(existing[1:]):
                    for c2 in clusters[e2]:
                        clusters[existing[0]].append(c2)
                    del clusters[e2]
            clusters[existing[0]].append(seqs_sorted[0])

        else:
            clusters.append([seqs_sorted[0]])
        del seqs_sorted[0]
        del neighbors_sorted[0]
    
    df=pd.DataFrame({"members":clusters,"repr_accession":[x[0] for x in clusters],"entry_id":["c"+str(x) for x in range(len(clusters))]}).set_index("entry_id")
    if(exploded):
        return explode_clusters_df(df)
    else:
        return df
