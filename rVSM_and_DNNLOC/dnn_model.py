from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy
from sklearn.neural_network import MLPRegressor
import joblib
from joblib import Parallel, delayed, cpu_count
from ray.util.joblib import register_ray
from math import ceil
import numpy as np
import os
import ray
import joblib

register_ray()

def oversample(samples):
    """ Oversamples the features for label "1" 
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    samples_ = []

    # oversample features of buggy files
    for i, sample in enumerate(samples):
        samples_.append(sample)
        if i % 51 == 0:
            for _ in range(9):
                samples_.append(sample)

    return samples_


def features_and_labels(samples):
    """ Returns features and labels for the given list of samples
    
    Arguments:
        samples {list} -- samples from features.csv
    """
    features = np.zeros((len(samples), 5))
    labels = np.zeros((len(samples), 1))

    for i, sample in enumerate(samples):
        features[i][0] = float(sample["rVSM_similarity"])
        features[i][1] = float(sample["collab_filter"])
        features[i][2] = float(sample["classname_similarity"])
        features[i][3] = float(sample["bug_recency"])
        features[i][4] = float(sample["bug_frequency"])
        labels[i] = float(sample["match"])

    return features, labels


def kfold_split_indexes(k, len_samples):
    """ Returns list of tuples for split start(inclusive) and 
        finish(exclusive) indexes.
    
    Arguments:
        k {integer} -- the number of folds
        len_samples {interger} -- the length of the sample list
    """
    step = ceil(len_samples / k)
    ret_list = [(start, start + step) for start in range(0, len_samples, step)]

    return ret_list


def kfold_split(bug_reports, samples, start, finish):
    """ Returns train samples and bug reports for test
    
    Arguments:
        bug_reports {list of dictionaries} -- list of all bug reports
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
    """
    train_samples = samples[:start] + samples[finish:]
    test_samples = samples[start:finish]

    test_br_ids = set([s["report_id"] for s in test_samples])
    test_bug_reports = [br for br in bug_reports if br["id"] in test_br_ids]

    return train_samples, test_bug_reports


def train_dnn(
    project, i, num_folds, samples, start, finish, sample_dict, bug_reports, br2files_dict
):
    """ Trains the dnn model and calculates top-k accuarcies
    
    Arguments:
        i {interger} -- current fold number for printing information
        num_folds {integer} -- total fold number for printing information
        samples {list} -- samples from features.csv
        start {integer} -- start index for test fold
        finish {integer} -- start index for test fold
        sample_dict {dictionary of dictionaries} -- a helper collection for fast accuracy calculation
        bug_reports {list of dictionaries} -- list of all bug reports
        br2files_dict {dictionary} -- dictionary for "bug report id - list of all related files in features.csv" pairs
    """
    print("Fold: {} / {}".format(i + 1, num_folds), end="\r")

    train_samples, test_bug_reports = kfold_split(bug_reports, samples, start, finish)
    train_samples = oversample(train_samples)
    np.random.shuffle(train_samples)
    X_train, y_train = features_and_labels(train_samples)
    model_path = f"models/replication/{project}_{i + 1}"
    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )

    with joblib.parallel_backend('ray'):
        clf.fit(X_train, y_train.ravel())
    
    joblib.dump(clf,  model_path, compress=9)

    acc_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=clf)

    return acc_dict


def dnn_model_kfold(project, k=10):
    """ Run kfold cross validation in parallel
    
    Keyword Arguments:
        k {integer} -- the number of folds (default: {10})
    """
    samples = csv2dict(f"../data/{project}/features.csv")

    br_file_path = f"../data/{project}/{project}.txt"
    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples, br_file_path)

    # if project == "Eclipse_Platform_UI":
    #     for i, j in br2files_dict.items():
    #         br2files_dict[i] = [k[8:] for k in j]

    sample_dict, bug_reports, br2files_dict = ray.put(sample_dict), ray.put(bug_reports), ray.put(br2files_dict)
    

    np.random.shuffle(samples)
    samples_len = len(samples)
    samples = ray.put(samples) 

    model_path = f"models/replication/{project}"
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok = True)

    @ray.remote
    def parallelize(project, i, k, samples, start, step, sample_dict, bug_reports, br2files_dict):
        return train_dnn(project, i, k, samples, start, step, sample_dict, bug_reports, br2files_dict)

    # K-fold Cross Validation in parallel
    # register_ray()
    # with joblib.parallel_backend('ray'):
    # metrics = Parallel(n_jobs=-2)(  # Uses all cores but one
    #     delayed(train_dnn)(
    #         i, k, samples, start, step, sample_dict, bug_reports, br2files_dict
    #     )
    #     for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    # )
    metrics = []
    for i, (start, step) in enumerate(kfold_split_indexes(k, (samples_len))):
        metrics.append(parallelize.remote(project, i, k, samples, start, step, sample_dict, bug_reports, br2files_dict))
    metrics = ray.get(metrics)
    # print(metrics)
        # Calculating the average accuracy from all folds
    avg_acc_dict = {}
    for key in metrics[0][0].keys():
        avg_acc_dict[key] = round(sum([d[key] for (d, _, _) in metrics]) / len(metrics), 3)

    avg_map = sum([map_score for ( _, map_score, _ ) in metrics]) / len(metrics)
    avg_mrr = sum([mrr_score for ( _, _, mrr_score ) in metrics]) / len(metrics)

    return {
        "accuracy": avg_acc_dict,
        "avg_mrr": avg_mrr,
        "avg_map": avg_map
    }




def bl_dataset_train_dnn(data_path, project):
   

    train_samples = csv2dict(f"{data_path}/{project}/{project}_train_features.csv")
    train_br_file_path = f"{data_path}/{project}/{project}_train_br.csv"
    train_sample_dict, train_bug_reports, train_br2files_dict = helper_collections(train_samples, train_br_file_path)



    bl_data_path = f"/home/varumuga/scratch/Thesis/bench_bl_dataset/bl_dataset/within_project"
    test_samples = csv2dict(f"{bl_data_path}/{project}/{project}_test_features.csv")
    test_br_file_path = f"{bl_data_path}/{project}/{project}_test_br.csv"
    test_sample_dict, test_bug_reports, test_br2files_dict = helper_collections(test_samples, test_br_file_path)


    X_train, y_train = features_and_labels(train_samples)

    clf = MLPRegressor(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(300,),
        random_state=1,
        max_iter=10000,
        n_iter_no_change=30,
    )

    with joblib.parallel_backend('ray'):
        clf.fit(X_train, y_train.ravel())

    data_setting =  os.path.basename(os.path.normpath(data_path))
    joblib.dump(clf,  f"models/{project}/{data_setting}", compress=9)
    train_acc_dict, train_map_value, train_mrr_value = topk_accuarcy(train_bug_reports, train_sample_dict, train_br2files_dict, clf=clf)
    test_acc_dict, test_map_value, test_mrr_value = topk_accuarcy(test_bug_reports, test_sample_dict, test_br2files_dict, clf=clf)

    train_metrics = {
        "accuracy": train_acc_dict,
        "avg_mrr": train_mrr_value,
        "avg_map": train_map_value
    }

    test_metrics = {
        "accuracy": test_acc_dict,
        "avg_mrr": test_mrr_value,
        "avg_map": test_map_value
    }
    return train_metrics, test_metrics