# %%
from __future__ import division

# import ray
# import modin.pandas as ray_pd
import pandas as ray_pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
from os import listdir
from os.path import isfile, join
import math
import time 

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.test.utils import get_tmpfile
from gensim import similarities
from gensim import models

import warnings
import multiprocessing
from tqdm import tqdm_notebook
from time import gmtime, strftime
from ast import literal_eval

import zlib

from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score
import json

# %%
# ray.init()

# %%
def evaluate_helper(ranked_files,fixes):
    """
    @Receives: list of ranked files(which is predicted by algorithm) and the ground truth(fixes)
    @Process: This is a function aimed to help evaluate, and calculates the AP and first_pos 
    @Returns: A dictionary containing the AP(average precision) and first_pos (first retrieved fix file)
    """
    found=0
    first_pos=-1
    average_precision=0
    for i,predictionFix in enumerate(ranked_files):
        for actualFix in fixes:
            if actualFix in predictionFix:
                if first_pos==-1:
                    first_pos=i+1
                found+=1
                average_precision+=found/(i+1)        

    AP=average_precision/found if found>0 else 0
    return {"AP":AP,"first_pos":first_pos}


def evaluate(all_bugs_df,source_codes_df):
    """
    @Receives: The main dataframe
    @Process: Evaluates the predicted files for each bugreport in all_bugs_df
    @Returns: MAP and MRR calculated from eligible bugreports(the ones with
    at least one fix file in this version of code) in the dataframe and number of eligible bug reports.
    """
    all_results=[]
    top_founds=[]
    average_precisions=[]
    topk_counters = [0] * 20
    mrr_score = []
    map_scores = []
    top_k_counter = [0] * 20
    for i,br in all_bugs_df.iterrows():
#         a = (not source_codes_df.loc[source_codes_df.filename.apply(lambda filename: any(fix in filename for fix in br['fix']))].empty)
#         b = (not source_codes_df.loc[source_codes_df.filename.str.contains(" ".join(br["fix"]))].empty)
#         if a != b:
#             print(a, b)
#             print(br["fix"])
#             raise Exception()
        pool_sc_df = source_codes_df[(source_codes_df.report_id == br.id) & (source_codes_df.project_name == br.project_name)]
        # print('1' in pool_sc_df.match)
        # print(pool_sc_df.match)
        # print('1' in pool_sc_df.match.values, pool_sc_df.match)
#         if '1' in pool_sc_df.match.values: 
#         # if not source_codes_df.loc[source_codes_df.filename.apply(lambda filename: any(fix in filename for fix in br['fix']))].empty:
# #         if not source_codes_df.loc[source_codes_df.filename.str.contains(" ".join(br["fix"]))].empty:
#             predicted_files=br['total_score'].keys()
#             result=evaluate_helper(predicted_files,br['fix'])
#             top_founds.append(result['first_pos'])
#             average_precisions.append(result['AP'])
#             all_results.append(result)
#         else:
#             top_founds.append(-1)
#             average_precisions.append(0.0)
        
        true_label = pool_sc_df["match"].astype('int')
        predicted = pool_sc_df["sim_score"]
        map_scores.append(average_precision_score(true_label, predicted))
    
        sorted_prediction_rank = np.argsort(-predicted)
        sorted_prediction_value = np.array([true_label.iloc[item] for item in sorted_prediction_rank])
        lowest_retrieval_rank = (sorted_prediction_value == 0).argsort(axis=0)
        mrr_score.append(1.0 / (lowest_retrieval_rank[0] + 1))

        sorted_label_rank = np.argsort(-true_label)
        for position_k in range(0, 20):
            common_item = np.intersect1d(sorted_prediction_rank[:(position_k + 1)],
                                         sorted_label_rank[:(position_k + 1)])
            if len(common_item) > 0:
                top_k_counter[position_k] += 1
        # all_bugs_df["top_found"]=top_founds
    # all_bugs_df["average_precision"]=average_precisions
    
    #Calculating the MAP and MRR
    # MAP,MRR=(0,0)
    # if len(all_results)>0:
    #     for result in all_results:
    #         MAP+=result['AP']
    #         MRR+=1/result['first_pos'] if result['first_pos']>0 else 0
    #     MAP/=len(all_results)
    #     MRR/=len(all_results)
    acc_dict = {}
    for i, counter in enumerate(top_k_counter):
        acc = counter / (all_bugs_df.shape[0])
        acc_dict[i + 1] = round(acc, 3)
    

    print("eligible_br_count: ",len(all_results))
    return (np.mean(map_scores),np.mean(mrr_score),len(all_results), acc_dict)


# %%
def getNormValue(x,maximum,minimum):
    return 6*((x - minimum)/(maximum - minimum))

def getLenScore(length):
    return (math.exp(length) / (1 + math.exp(length)))

def calulateLengthScore(source_codes_df):
    """
    Receives: a list of sizes of codes and the index
    Process: calculate a boost score for the specified index based on length of that code
    Returns: length boosting score 
    """
    average_size=source_codes_df['size'].mean()
    standard_deviation=source_codes_df['size'].std() 
    low=average_size-3*standard_deviation
    high= average_size+3*standard_deviation
    minimum=int(low) if low>0 else 0
        
    len_scores=[]
    for i,eachLen in source_codes_df['size'].items():
        score=0
        nor=getNormValue(eachLen,high,minimum)
        if eachLen!=0:
            if eachLen>=low and eachLen<=high:
                score=getLenScore(nor)
            elif eachLen<low:
                score=0.5
            elif eachLen>high:
                score = 1.0
        len_scores.append(score)
    source_codes_df['lengthScore']=len_scores

    return source_codes_df
    
def inverse_doc_freq(idf,D):
    return math.log(D/idf)

def term_freq(tf_list):
    return [(math.log(tf+1)) for tf in tf_list]

def np_normalizer(arr):
    """
    @Receives: a list of numbers
    @Process: normalizes all the values and map them to range of [0,1]
    @Returns: list of normalized numbers
    """
    if len(arr)>0:
        maximum=np.amax(arr)
        minimum=np.amin(arr)
        if maximum!=minimum:
            return (arr-minimum)/(maximum-minimum)
    return arr

def normalizer(Dict):
    """
    @Receives: a list of numbers
    @Process: normalizes all the values and map them to range of [0,1]
    @Returns: list of normalized numbers
    """
    if len(Dict)>0:
        maximum=max(Dict.items(), key=operator.itemgetter(1))[1]
        minimum=min(Dict.items(), key=operator.itemgetter(1))[1]
        for key,value in Dict.items():
            if maximum!=minimum:
                Dict[key]=(value-minimum)/(maximum-minimum)
            else:
                Dict[key]=1.0
    return Dict
    

def TFIDF_transform(all_bugs_df,source_codes_df):

    dictionary = gensim.corpora.Dictionary(list(source_codes_df['code']))
    corpus = [dictionary.doc2bow(doc) for doc in list(source_codes_df['code'])]
    tfidf_weights = models.TfidfModel(corpus,wlocal=term_freq,wglobal=inverse_doc_freq,normalize=False)
    return tfidf_weights[corpus], all_bugs_df.text.apply(lambda x: tfidf_weights[dictionary.doc2bow(x)])

def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

# %%

def build_Doc2Vec_models(vec_size,alpha,window_size,run_name,all_bugs_df=None,source_codes_df=None, project=None):
    """
    Process: 1- Loads all the bug reports from all the group/projects in Data directory
             2- Makes a Doc2Vec model and trains it based on all the bugreports
    Returns: Trained model
    """
    print("\n\t Now building the Combined Doc2Vec model ... ")
    model_folder = os.path.join(os.getcwd(),'Models')
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    
    project_model_path=os.path.join(model_folder, run_name, project) 
    if not os.path.exists(project_model_path):
        os.makedirs(project_model_path)
        
    Path(project_model_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Writing model for {project} to {project_model_path}")
    # project_dbow_model_path=os.path.join(os.getcwd(),'Models','combined_doc2vec_model_dbow')
    dmm_model_path = os.path.join(project_model_path, 'combined_doc2vec_model_dmm')
    dbow_model_path = os.path.join(project_model_path, 'combined_doc2vec_model_dbow')
    
    fname_dmm = get_tmpfile(dmm_model_path)
    fname_dbow = get_tmpfile(dbow_model_path)
    
    if os.path.isfile(dmm_model_path) and os.path.isfile(dbow_model_path):
        revectorize=False
        model_dmm = Doc2Vec.load(fname_dmm)
        model_dbow = Doc2Vec.load(fname_dbow)
        print("*** Combined Doc2Vec Model is Loaded. ***")            
    else:
        revectorize=True
        documents = [TaggedDocument(all_bugs_df.iloc[i].text, [i]) for i in range(len(all_bugs_df))]
        documents = documents + [TaggedDocument(source_codes_df.iloc[i].code, [len(all_bugs_df)+i]) for i in range(len(source_codes_df))]

        model_dmm = Doc2Vec(vector_size=vec_size, window=window_size, min_count=2,
                        workers=multiprocessing.cpu_count(),
                        alpha=alpha, min_alpha=alpha/2,dm=1)
        model_dmm.build_vocab(documents)
        model_dmm.train(documents,total_examples=model_dmm.corpus_count,epochs=20)
#         model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model_dmm.save(fname_dmm)
        
        model_dbow = Doc2Vec(dm=0, vector_size=vec_size, negative=5,
                             hs=0,min_count=2, sample = 0, workers=multiprocessing.cpu_count(),
                             alpha=alpha, min_alpha=alpha/3)
        model_dbow.build_vocab(documents)
        model_dbow.train(documents,total_examples=model_dbow.corpus_count,epochs=20)
#         model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model_dbow.save(fname_dbow)
        print("*** Combined Doc2Vec Model is Trained. ***")
    concatinated_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    print(">> Size of Vocabulary is: {}".format(len(model_dmm.wv)))
    print(">> Number of whole Documents: {}".format(model_dmm.corpus_count))
    
    return (concatinated_model,revectorize)


def synthesize(sourceCodeScores,bugReportScores):
    sourceCodeScores=normalizer(sourceCodeScores)
    bugReportScores=normalizer(bugReportScores)
    for file in bugReportScores.keys():
        if file in sourceCodeScores.keys():
            sourceCodeScores[file]=sourceCodeScores[file]*0.8+bugReportScores[file]*0.2
    return sourceCodeScores

def combine(scoring_one, scoring_two):
    scoring_one=np_normalizer(scoring_one)  
    scoring_two=np_normalizer(scoring_two)
    result=scoring_one+scoring_two
    return result
# %%
# @ray.remote
def find_similarity(index, doc2vec_index, all_bugs_doc2vec, source_codes_df, all_bugs_df, direct_tfidf_index ,direct_WE_index, indirect_tfidf_index, indirect_WE_index):
    try:
        direct_doc2vec_similarities=cos_matrix_multiplication(doc2vec_index, all_bugs_doc2vec.tfidf_vector)
        direct_tfidf_similarities=direct_tfidf_index[all_bugs_doc2vec.tfidf_vector]
        direct_similarities=combine(direct_tfidf_similarities,direct_doc2vec_similarities)
        sc_scores = direct_similarities * source_codes_df.lengthScore.values
        sourceCodeScores={source_codes_df.iloc[j].filename.split('.')[-2]: (direct_similarities[j])*source_codes_df.iloc[j].lengthScore 
                                                  for j in range(len(source_codes_df))
                                                  if len(source_codes_df.iloc[j].filename.split('.'))>1}
#         sourceCodeScores={source_codes_df.iloc[j].filename: (doc2vec_similarities[j])*source_codes_df.iloc[j].lengthScore 
#                                           for j in range(len(source_codes_df))}
        
        indirect_doc2vec_similarities=cos_matrix_multiplication(indirect_WE_index, all_bugs_doc2vec.doc2vec_vector)
        indirect_tfidf_similarities=indirect_tfidf_index[all_bugs_doc2vec.tfidf_vector]
        indirect_similarities=combine(indirect_tfidf_similarities,indirect_doc2vec_similarities)
        bugReportScores=dict({})
        
        for j,(idx,other_br) in enumerate(all_bugs_df.iterrows()):
            for fixFile in other_br.fix:
                if idx != index:
                    if fixFile.split('.')[-2] in bugReportScores.keys():
                        if indirect_similarities[j]>=bugReportScores[fixFile.split('.')[-2]]:
                            bugReportScores[fixFile.split('.')[-2]]=indirect_similarities[j]
                    else:
                        bugReportScores[fixFile.split('.')[-2]]=indirect_similarities[j]
                            
        ranking=synthesize(sourceCodeScores,bugReportScores)

        return (index, {file:score for file,score in sorted(ranking.items(),key=lambda tup: tup[1],reverse=True)}, sc_scores)
    except Exception as e:
#         logging.error(traceback.format_exc())
        return (index, {}, 0)    

# %%
# @ray.remote
def localize(project, source_codes_df, all_bugs_df, df_type):
    print(f"Localizing Now ... for project: {project} on {df_type} data")
    start = time.time()
    source_codes_df=calulateLengthScore(source_codes_df)        
    doc2vec_index=np.array(list(source_codes_df.doc2vec_vector))
    results = []
    scores = []
    source_codes_df=calulateLengthScore(source_codes_df)
    
    direct_tfidf_index = similarities.SparseMatrixSimilarity(list(source_codes_df.tfidf_vector),num_features=BugLocalizer.dictionary_length)
    direct_WE_index=np.array(list(source_codes_df.doc2vec_vector))
    
    indirect_tfidf_index = similarities.SparseMatrixSimilarity(list(all_bugs_df.tfidf_vector),num_features=BugLocalizer.dictionary_length)
    indirect_WE_index=np.array(list(all_bugs_df.doc2vec_vector))

    for index, (i, br) in enumerate(all_bugs_df.iterrows()):
        sc_codes_for_br = source_codes_df[(source_codes_df.report_id == br.id) & (source_codes_df.project_name == br.project_name)]
        scores.append({})
        idx_, files_score_map, sc_scores = find_similarity(index, doc2vec_index[sc_codes_for_br.index], br, sc_codes_for_br, all_bugs_df, direct_tfidf_index ,direct_WE_index, indirect_tfidf_index, indirect_WE_index)
        source_codes_df["sim_score"].loc[(source_codes_df.report_id == br.id) & (source_codes_df.project_name == br.project_name)] = sc_scores
        results.append((idx_, files_score_map))
    for res in results:
        try:
            idx, process_result = res #ray.get(res)
            scores[idx] = process_result
        except Exception as e:
            print(e)
            pass          
    all_bugs_df['total_score']=scores
    end = time.time()
    print(f"Localizing done in {end - start} for {project} on {df_type} dat")

# %%
def to_csv(data_type, project, bugreports_df, source_codes_df, resultPath):
    bug_reports_path=os.path.join(resultPath, f'BugReports/{data_type}') 
    sc_path=os.path.join(resultPath,f'SourceFiles/{data_type}')
    if not os.path.exists(bug_reports_path):
        os.makedirs(bug_reports_path)
    if not os.path.exists(sc_path):
        os.makedirs(sc_path)

    result_Bug_file=os.path.join(bug_reports_path,project+"_BugReports.csv")
    result_source_file=os.path.join(sc_path,project+"_SourceFiles.parquet")
    bugreports_df.to_csv(result_Bug_file)
    if  len(source_codes_df)<100:
        source_codes_df.to_csv(result_source_file)

# %%
def write_result(data_type, result, result_path, project):
    write_to_result_path = os.path.join(result_path,data_type)
    print("Writing results to", write_to_result_path)
    if not os.path.exists(write_to_result_path):
        os.makedirs(write_to_result_path)
    
    group_result=open(os.path.join(write_to_result_path,"results_{}.csv".format(project)),'w')
    group_result.write("project , MAP , MRR , #ofBugReports\n")
    group_result.write(project+','+str(result[0])+','+str(result[1])+','+str(result[2])+"\n")
    group_result.close()
    
    map_score, mrr_score, _, acc_k = result
    acc_k["map"] = map_score
    acc_k["mrr"] = mrr_score
    
    with open(os.path.join(write_to_result_path,"results_{}.txt".format(project)), 'w') as f:
        json.dump(acc_k, f)

# %%
def source_code_decompress(compressed_sc):
    sc_code = zlib.decompress(bytes.fromhex(compressed_sc)).decode()
    return sc_code.split("-|-") 

# %%
class BugLocalizer:    
    
    def __init__(self,project,result_path, data_folder,run_name):
        self.project=project
        self.resultPath=result_path
        self.dataFolder = data_folder
        if not os.path.exists(self.dataFolder):
            os.makedirs(self.dataFolder)
        self.combined_Doc2vec=None
        
        self.train_source_codes=ray_pd.DataFrame([])
        self.train_bugreports=ray_pd.DataFrame([])
        # self.train_bugreports_doc2vec = None
        # self.train_source_codes_doc2vec = None

        self.test_source_codes=ray_pd.DataFrame([])
        self.test_bugreports=ray_pd.DataFrame([])
        # self.test_bugreports_doc2vec = None
        # self.test_source_codes_doc2vec = None

        self.vectorize=False
        
        self.loadEverything()
        
        self.combined_Doc2vec, _ =build_Doc2Vec_models(vec_size=100,alpha=0.045,window_size=5,
                                                    run_name=run_name,
                                               all_bugs_df=self.train_bugreports,
                                               source_codes_df=self.train_source_codes, project=self.project)
        
        
        self.vectorizeSourceCodes()
        self.vectorizeBugreports()
        self.tfidf_sc_br()
        # dataFolder=""
            
    def execute(self):
        print("\t ****** Localizing Bugs for project: {} ******".format(self.project))
        revectorize=False
        self.localize()
        self.evaluate()
        self.to_csv()
        self.write_result()
        
    def loadEverything(self):
        
        project_folder = os.path.join(self.dataFolder, self.project)
        
        train_bugReportFile=os.path.join(project_folder,f'{self.project}_train_br.pickle')
        
        test_bugReportFile=os.path.join(project_folder,f'{self.project}_test_br.pickle')
        
        self.train_bugreports = ray_pd.read_pickle(train_bugReportFile)
        
        self.test_bugreports = ray_pd.read_pickle(test_bugReportFile)
        
        self.train_bugreports = self.train_bugreports.rename(columns={'files': 'fix', 'report_processed': 'text'}) 

        self.train_bugreports = self.train_bugreports.reset_index()
        
        self.test_bugreports = self.test_bugreports.rename(columns={'files': 'fix', 'report_processed': 'text'}) 

        self.test_bugreports = self.test_bugreports.reset_index()
        
        train_sc_file=os.path.join(project_folder,f'{self.project}_train_sc')
        
        train_sc_csv_file = os.path.join(project_folder,f'{self.project}_train_sc.csv')
        
        print(f"Loading train sc from {train_sc_file}")
        
        if os.path.isfile(train_sc_csv_file):
            self.train_source_codes = ray_pd.read_csv(train_sc_csv_file)
        else:
            self.train_source_codes = ray_pd.read_parquet(train_sc_file)

        
        test_sc_file=os.path.join(project_folder,f'{self.project}_test_sc')
        
        test_sc_csv_file = os.path.join(project_folder,f'{self.project}_test_sc.csv')
            
        print(f"Loading test sc from {test_sc_file}")
        
        if os.path.isfile(test_sc_csv_file):
            self.test_source_codes = ray_pd.read_csv(test_sc_csv_file)
        else:
            self.test_source_codes = ray_pd.read_parquet(test_sc_file)
        
        self.train_source_codes = self.train_source_codes.rename(columns={'file': 'filename'}) 
        
        self.test_source_codes = self.test_source_codes.rename(columns={'file': 'filename'})
        
        self.train_source_codes["code"] = self.train_source_codes.file_content_processed.apply(source_code_decompress)
        
        self.test_source_codes["code"] = self.test_source_codes.file_content_processed.apply(source_code_decompress)
        
        self.train_source_codes["size"] = ray_pd.to_numeric(self.train_source_codes["size"])
        
        self.test_source_codes["size"] = ray_pd.to_numeric(self.test_source_codes["size"])
        
        self.test_source_codes["sim_score"] = 0
        self.train_source_codes["sim_score"] = 0


        self.test_source_codes = self.test_source_codes.reset_index()

        self.train_source_codes = self.train_source_codes.reset_index()
#         ray.put(self.train_source_codes)
        
#         ray.put(self.test_source_codes)
        print("Train Bug Report size:" , self.train_bugreports.shape)
        print("Train Source Code size:" , self.train_source_codes.shape)
        
        print("Test Bug Report size:" , self.test_bugreports.shape)
        print("Test Source Code size:" , self.test_source_codes.shape)
        print("Both Train and Test Data Loaded")
        
        # self.group=self.all_bugs_df["group"].iloc[0]
    def tfidf_sc_br(self):
        project_folder = os.path.join(self.dataFolder, self.project)

        train_projects_br_tfidf_file = os.path.join(project_folder, f'{self.project}_train_br_tfidf.npy')
        train_projects_sc_tfidf_file = os.path.join(project_folder, f'{self.project}_train_sc_tfidf.npy')
        if not os.path.isfile(train_projects_br_tfidf_file):
            sc_tfidf, br_tfidf = TFIDF_transform(all_bugs_df=self.train_bugreports,source_codes_df=self.train_source_codes)
            br_tfidf = [i for i in br_tfidf]
            sc_tfidf = [i for i in br_tfidf]
            np.save(train_projects_br_tfidf_file, np.array(br_tfidf))
            np.save(train_projects_sc_tfidf_file, np.array(sc_tfidf))
        else:
            br_tfidf = [i for i in np.load(train_projects_br_tfidf_file).tolist()]
            sc_tfidf = [i for i in np.load(train_projects_sc_tfidf_file).tolist()]
        self.train_bugreports["tfidf_vector"] = br_tfidf
        self.train_source_codes["tfidf_vector"] = sc_tfidf
        
    def vectorizeBugreports(self):
        
        project_folder = os.path.join(self.dataFolder, self.project)
        
        train_projects_bugreports_doc2vec_file = os.path.join(project_folder, f'{self.project}_train_br_doc2vec.npy')
        if not os.path.isfile(train_projects_bugreports_doc2vec_file):
            train_br_doc2vec = [self.combined_Doc2vec.infer_vector(i) for i in self.train_bugreports.text.tolist()]
            np.save(train_projects_bugreports_doc2vec_file, np.array(train_br_doc2vec))

        else:
            train_br_doc2vec = [i for i in np.load(train_projects_bugreports_doc2vec_file).tolist()]
        self.train_bugreports["doc2vec_vector"] = train_br_doc2vec
        
        test_projects_bugreports_doc2vec_file = os.path.join(project_folder, f'{self.project}_test_br_doc2vec.npy')
        if not os.path.isfile(test_projects_bugreports_doc2vec_file):
            test_br_doc2vec = [self.combined_Doc2vec.infer_vector(i) for i in self.test_bugreports.text.tolist()]
            np.save(test_projects_bugreports_doc2vec_file, np.array(test_br_doc2vec))

        else:
            test_br_doc2vec = [i for i in np.load(test_projects_bugreports_doc2vec_file).tolist()]
        self.test_bugreports["doc2vec_vector"] = test_br_doc2vec
            # BugLocalizer.all_projects_bugreports["doc2vec_vector"] = all_projects_bugreports_doc2vec
#             BugLocalizer.all_projects_bugreports_doc2vec = all_projects_bugreports_doc2vec

    def vectorizeSourceCodes(self):
        project_folder = os.path.join(self.dataFolder, self.project)
        train_projects_sc_doc2vec_file = os.path.join(project_folder, f'{self.project}_train_sc_doc2vec.npy')
        if not os.path.isfile(train_projects_sc_doc2vec_file):
            
            train_sc_doc2vec = [self.combined_Doc2vec.infer_vector(i) for i in self.train_source_codes.code.tolist()]
            np.save(train_projects_sc_doc2vec_file, np.array(train_sc_doc2vec))

        else:
            train_sc_doc2vec = [i for i in np.load(train_projects_sc_doc2vec_file).tolist()]
        self.train_source_codes["doc2vec_vector"] = train_sc_doc2vec


        test_projects_sc_doc2vec_file = os.path.join(project_folder, f'{self.project}_test_sc_doc2vec.npy')
        if not os.path.isfile(test_projects_sc_doc2vec_file):
            
            test_sc_doc2vec = [self.combined_Doc2vec.infer_vector(i) for i in self.test_source_codes.code.tolist()]
            np.save(test_projects_sc_doc2vec_file, np.array(test_sc_doc2vec))

        else:
            test_sc_doc2vec = [i for i in np.load(test_projects_sc_doc2vec_file).tolist()]
        self.test_source_codes["doc2vec_vector"] = test_sc_doc2vec

        
            # BugLocalizer.all_projects_source_codes["doc2vec_vector"] = all_projects_source_codes_doc2vec
#             BugLocalizer.all_projects_source_codes_doc2vec = all_projects_source_codes_doc2vec

    def localize(self):
        (localize(self.project, self.train_source_codes, self.train_bugreports, "train"))

        (localize(self.project, self.test_source_codes, self.test_bugreports, "test"))

    def evaluate(self):
        self.train_result=evaluate(self.train_bugreports,self.train_source_codes)
        print("Train Result/"+self.project+":\n\t",'*'*4," MAP: ",self.train_result[0],'*'*4,'\n\t','*'*4," MRR: ",self.train_result[1],'*'*4,"\n","-"*100)


        self.test_result=evaluate(self.test_bugreports,self.test_source_codes)
        print("Test Result/"+self.project+":\n\t",'*'*4," MAP: ",self.test_result[0],'*'*4,'\n\t','*'*4," MRR: ",self.test_result[1],'*'*4,"\n","-"*100)

    def to_csv(self):
        to_csv("train", self.project, self.train_bugreports, self.train_source_codes, self.resultPath)
        to_csv("test", self.project, self.test_bugreports, self.test_source_codes, self.resultPath)
        
    def write_result(self):
        write_result("train", self.train_result, self.resultPath, self.project)

        write_result("test", self.test_result, self.resultPath, self.project)
        

# %%
# @ray.remote
def do_2(data_folder, project, result_path):
    core=BugLocalizer(project=project,result_path=result_path, data_folder=data_folder)
#     print("Hello")
    core.execute()
    return core

# %%
def folder_structure(run_name):
    result_path=os.path.join(os.getcwd(),"Result",run_name,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path
    

run_name='cross_project_and_within_project'
result_path=folder_structure(run_name)
dataFolder = os.path.join(os.getcwd(),f'/home/varumuga/scratch/Thesis/replications/globug-replication-graham/globug_bl_dataset/new_dataset/{run_name}')
# BugLocalizer.dataFolder=dataFolder
# BugLocalizer.loadEverything()

#'AspectJ'
all_projects= ['AspectJ', 'Tomcat', 'SWT', 'JDT', 'Birt', 'Eclipse_Platform_UI']
executed = []
for proj in all_projects:
#     core=BugLocalizer.remote(project=project,result_path=result_path)
#         print(dataFolder, proj.strip(), result_path)
    # do_2(dataFolder, proj.strip(), result_path)
    # executed.append(do_2.remote(dataFolder, proj.strip(), result_path))
    start = time.time()
    core=BugLocalizer(project=proj,result_path=result_path, data_folder=dataFolder,run_name=run_name)
    #     print("Hello")
    core.execute()
    print(f"{proj} completed in", time.time() - start)
    
# ray.get(executed)

# %%
base_path = os.path.basename(os.path.normpath(result_path))

method=run_name
runNumber=base_path
def create_result(data_type):
    all_results_csv=[os.path.join(os.getcwd(),"Result",method,runNumber,data_type,folder) 
                     for folder in listdir(os.path.join(os.getcwd(),"Result",method,runNumber,data_type)) if '.csv' in folder]
    results_df=ray_pd.DataFrame([])
    for result_csv in all_results_csv:
        res=ray_pd.read_csv(result_csv,index_col=[0],header=0)
        results_df=results_df.append(res)
    results_df.to_csv(os.path.join(os.getcwd(),"Result",method,f'{data_type}_result.csv'))

create_result("train")
create_result("test")
# project_size_df=pd.read_csv('project_src_size.csv',index_col=[0],header=0)
# results_df
# results_df=pd.merge(results_df, project_size_df,
#                                       left_index=True,
#                                       right_index=True)
# results_df=results_df.reset_index()
# results_df=results_df.set_index(' #ofBugReports')



# %%



