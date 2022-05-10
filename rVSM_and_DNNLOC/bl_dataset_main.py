from dnn_model import bl_dataset_train_dnn
from rvsm_model import rsvm_model, bl_dataset_rsvm_model
import time as time
import os
import json
import argparse

from time import gmtime, strftime

def dnn(project, run_method, data_folder, run_start_time):
    start = time.time()
    train_dump_file_folder = f"Results/{run_start_time}/{run_method}/{project}/train"
    test_dump_file_folder = f"Results/{run_start_time}/{run_method}/{project}/test"

    if not os.path.isdir(train_dump_file_folder):
        os.makedirs(train_dump_file_folder, exist_ok = True)
    
    if not os.path.isdir(test_dump_file_folder):
        os.makedirs(test_dump_file_folder, exist_ok = True)

    model_path = f"models/{project}"
    if not os.path.isdir(model_path):
        os.makedirs(model_path, exist_ok = True)

    train_dump_file = os.path.join(train_dump_file_folder, "dnn_metrics.txt")
    test_dump_file = os.path.join(test_dump_file_folder, "dnn_metrics.txt")

    train_metrics, test_metrics = bl_dataset_train_dnn(data_folder, project)

    with open(train_dump_file, "w") as f:
        json.dump(train_metrics, f)

    with open(test_dump_file, "w") as f:
        json.dump(test_metrics, f)

    print(f"DNN Competed for {project} in {time.time() - start}")
    return True


def rsvm(project, run_method, data_folder, run_start_time):
    start = time.time()
    train_dump_file_folder = f"Results/{run_start_time}/{run_method}/{project}/train"
    test_dump_file_folder = f"Results/{run_start_time}/{run_method}/{project}/test"

    if not os.path.isdir(train_dump_file_folder):
        os.makedirs(train_dump_file_folder, exist_ok = True)
    
    if not os.path.isdir(test_dump_file_folder):
        os.makedirs(test_dump_file_folder, exist_ok = True)

    train_dump_file = os.path.join(train_dump_file_folder, "rvsm_metrics.txt")
    test_dump_file = os.path.join(test_dump_file_folder, "rvsm_metrics.txt")

    train_metrics, test_metrics = bl_dataset_rsvm_model(data_folder, project)

    with open(train_dump_file, "w") as f:
        json.dump(train_metrics, f)

    with open(test_dump_file, "w") as f:
        json.dump(test_metrics, f)


    print(f"RSVM Competed for {project} in {time.time() - start}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_method", default=None, type=str, help="")
    args = parser.parse_args()
    # 
    projects = ['AspectJ', 'Tomcat', 'SWT', 'JDT', 'Birt','Eclipse_Platform_UI']

    # , 'Tomcat', 'SWT', 'JDT', 'Birt','Eclipse_Platform_UI'
    projects = ['AspectJ']
    bl_data_path = f"/home/varumuga/scratch/Thesis/bench_bl_dataset/bl_dataset/{args.run_method}"
    print(f"Starting Modeling for {args.run_method}")
    run_start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print(f"run_start_time", run_start_time)
    result_path = f"Results/{run_start_time}/{args.run_method}"
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
        
    for project in projects:
        (dnn(project, args.run_method, bl_data_path, run_start_time))
        (rsvm(project, args.run_method, bl_data_path, run_start_time))