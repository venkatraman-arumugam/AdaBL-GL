from util import csv2dict, tsv2dict, helper_collections, topk_accuarcy
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
import numpy as np



def bl_dataset_rsvm_model(data_path, project):


    train_samples = csv2dict(f"{data_path}/{project}/{project}_train_features.csv")
    # rvsm_list = [float(sample["rVSM_similarity"]) for sample in train_samples]
    train_br_file_path = f"{data_path}/{project}/{project}_train_br.csv"
    # These collections are speed up the process while calculating top-k accuracy
    train_sample_dict, train_bug_reports, train_br2files_dict = helper_collections(train_samples, train_br_file_path, True)

    train_acc_dict, train_avg_map, train_avg_mrr = topk_accuarcy(train_bug_reports, train_sample_dict, train_br2files_dict)

    train_metrics = {
        "accuracy": train_acc_dict,
        "avg_mrr": train_avg_mrr,
        "avg_map": train_avg_map
    }



    test_samples = csv2dict(f"{data_path}/{project}/{project}_test_features.csv")
    # rvsm_list = [float(sample["rVSM_similarity"]) for sample in train_samples]
    test_br_file_path = f"{data_path}/{project}/{project}_test_br.csv"
    # These collections are speed up the process while calculating top-k accuracy
    test_sample_dict, test_bug_reports, test_br2files_dict = helper_collections(test_samples, test_br_file_path, True)

    test_acc_dict, test_avg_map, test_avg_mrr = topk_accuarcy(test_bug_reports, test_sample_dict, test_br2files_dict)

    test_metrics = {
        "accuracy": test_acc_dict,
        "avg_mrr": test_avg_mrr,
        "avg_map": test_avg_map
    }
    return train_metrics, test_metrics

def rsvm_model(project):
    samples = csv2dict(f"../data/{project}/features.csv")
    rvsm_list = [float(sample["rVSM_similarity"]) for sample in samples]
    br_file_path = f"../data/{project}/{project}.txt"
    # These collections are speed up the process while calculating top-k accuracy
    sample_dict, bug_reports, br2files_dict = helper_collections(samples, br_file_path, True)

    acc_dict, avg_map, avg_mrr = topk_accuarcy(bug_reports, sample_dict, br2files_dict)

    return {
        "accuracy": acc_dict,
        "avg_mrr": avg_mrr,
        "avg_map": avg_map
    }
