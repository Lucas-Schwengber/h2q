# Eval mAP using two strategies
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import json
import pandas as pd

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - B1 @ B2.T)
    return distH


def calc_euclideanDist(F1, F2):
    distE = np.sum( (F1-F2)**2, axis = 1)
    return distE


def calc_cosineDist(F1, F2):
    cossim = torch.nn.CosineSimilarity()
    eps = 1e-5
    return eps + (1 - cossim(torch.Tensor(F1), torch.Tensor(F2)))*(1 - 2*eps)


def one_hot_labels(labels, n=0):
    labels = labels.astype(int)
    u_labels = np.unique(labels)
    if n == 0:
        n = u_labels.size
    ohl = np.zeros((labels.size, n))
    for i, l in enumerate(u_labels):
        ohl[ labels == l, i ] += 1
    return ohl


def mAP(query_hashes, retrieval_hashes, query_labels, retrieval_labels):
    # query_hashes: {-1,+1}^{mxq}
    # retrieval_hashes: {-1,+1}^{nxq}
    # query_labels: {0,1}^{mxl} or numbers
    # retrieval_labels: {0,1}^{nxl} or numbers

    num_query = query_labels.shape[0]
    map = 0

    for iter in range(num_query):
        gnd = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(query_hashes[iter, :], retrieval_hashes)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query

    return map


def precision_at(k,is_rel_item):
    return sum(is_rel_item[:k])/k


def AP(is_relevant_item, number_of_retrieved_items):
    if number_of_retrieved_items == 0:
            return 0
    relevant_item_position = np.asarray(np.where(is_relevant_item[:number_of_retrieved_items] == 1)) + 1.0
    number_of_retrieved_relevant_items = relevant_item_position.shape[-1]
    count = np.linspace(1, number_of_retrieved_relevant_items, number_of_retrieved_relevant_items) 
    if number_of_retrieved_relevant_items != 0:
        AP = np.sum(count / relevant_item_position)/number_of_retrieved_relevant_items
    else:
        AP = 0
    return AP

def precision_and_recall(query_features, retrieval_features, query_labels, retrieval_labels, k_values=[1]):
    #Convert entries to numpy for compatibility

    query_features = np.array(query_features)
    retrieval_features = np.array(retrieval_features)
    query_features /= np.linalg.norm(query_features, axis=1)[:, None]
    retrieval_features /= np.linalg.norm(retrieval_features, axis=1)[:, None]
    query_labels = np.array(query_labels)
    retrieval_labels = np.array(retrieval_labels)
    query_hashes = np.sign(query_features).astype(int)
    retrieval_hashes = np.sign(retrieval_features).astype(int)

    num_query = query_labels.shape[0]

    precision = {k: 0 for k in k_values}
    recall = {k: 0 for k in k_values}

    for iter in tqdm(range(num_query)):
        is_relevant_item = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
        number_of_relevant_items = np.sum(is_relevant_item).astype(int)
        if number_of_relevant_items == 0:
            continue
        hamming_distances = calc_hammingDist(query_hashes[iter, :], retrieval_hashes)
        cosine_distances = calc_euclideanDist(query_features[iter, :], retrieval_features)
        indexes_sorted = np.argsort(10*hamming_distances + cosine_distances)
        hamming_distances = hamming_distances[indexes_sorted]
        is_relevant_item = is_relevant_item[indexes_sorted]

        for k in k_values:
            number_of_retrieved_items = k
            precision[k] += 1/num_query*np.sum(is_relevant_item[:k])/number_of_retrieved_items
            recall[k] += 1/num_query*np.sum(is_relevant_item[:k])/number_of_relevant_items

    return precision, recall

def mAP_at(query_hashes, retrieval_hashes, query_features, retrieval_features, query_labels, retrieval_labels, h=None, k=None):
    # query_hashes: {-1,+1}^{mxq}
    # retrieval_hashes: {-1,+1}^{nxq}
    # query_features: R^{mxq}
    # retrieval_features: R^{nxq}
    # query_labels: {0,1}^{mxl} or numbers
    # retrieval_labels: {0,1}^{nxl} or numbers
    # h: threshold for hamming distance radius
    # k: max number of items to return 

    #Convert entries to numpy for compatibility
    query_hashes = np.array(query_hashes)
    retrieval_hashes = np.array(retrieval_hashes)
    query_features = np.array(query_features)
    retrieval_features = np.array(retrieval_features)
    query_features /= np.linalg.norm(query_features, axis=1)[:, None]
    retrieval_features /= np.linalg.norm(retrieval_features, axis=1)[:, None]
    query_labels = np.array(query_labels)
    retrieval_labels = np.array(retrieval_labels)

    num_query = query_labels.shape[0]
    map = 0

    for iter in range(num_query):
        is_relevant_item = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
        number_of_relevant_items = np.sum(is_relevant_item).astype(int)
        if number_of_relevant_items == 0:
            continue
        hamming_distances = calc_hammingDist(query_hashes[iter, :], retrieval_hashes)
        cosine_distances = calc_euclideanDist(query_features[iter, :], retrieval_features)
        indexes_sorted = np.argsort(10*hamming_distances + cosine_distances)
        hamming_distances = hamming_distances[indexes_sorted]
        is_relevant_item = is_relevant_item[indexes_sorted]

        if k is not None:
            number_of_retrieved_items = k
        elif h is not None:
            number_of_retrieved_items = np.sum(hamming_distances<=h)
        else:
            number_of_retrieved_items = len(is_relevant_item)

        if number_of_retrieved_items == 0:
            continue

        #costly way to compute mAP
        #map_ = sum([precision_at(j,is_rel_item)*is_rel_item[j] for j in range(1,n_ret_items+1)])/n_rel_items

        relevant_item_position = np.asarray(np.where(is_relevant_item[:number_of_retrieved_items] == 1)) + 1.0
        number_of_retrieved_relevant_items = relevant_item_position.shape[-1]
        count = np.linspace(1, number_of_retrieved_relevant_items, number_of_retrieved_relevant_items) 
        if number_of_retrieved_relevant_items != 0:
            AP = np.sum(count / relevant_item_position)/number_of_retrieved_relevant_items
        else:
            AP = 0

        map = map + AP

    map = map / num_query

    return map


def mAP_at_many(query_features, retrieval_features, query_labels, retrieval_labels, h_values=[], k_values=[]):
    # query_hashes: {-1,+1}^{mxq}
    # retrieval_hashes: {-1,+1}^{nxq}
    # query_features: R^{mxq}
    # retrieval_features: R^{nxq}
    # query_labels: {0,1}^{mxl} or numbers
    # retrieval_labels: {0,1}^{nxl} or numbers
    # h: threshold for hamming distance radius
    # k: max number of items to return 

    #Convert entries to numpy for compatibility
    query_features = np.array(query_features)
    retrieval_features = np.array(retrieval_features)
    query_features /= np.linalg.norm(query_features, axis=1)[:, None]
    retrieval_features /= np.linalg.norm(retrieval_features, axis=1)[:, None]
    query_labels = np.array(query_labels)
    retrieval_labels = np.array(retrieval_labels)
    query_hashes = np.sign(query_features).astype(int)
    retrieval_hashes = np.sign(retrieval_features).astype(int)

    num_query = query_labels.shape[0]
    map = {}

    map["full"] = 0

    for h in h_values:
        map[f"h={h}"] = 0

    for k in k_values:
        map[f"k={k}"] = 0

    for iter in range(num_query):
        is_relevant_item = (np.dot(query_labels[iter, :], retrieval_labels.transpose()) > 0).astype(np.float32)
        number_of_relevant_items = np.sum(is_relevant_item).astype(int)
        if number_of_relevant_items == 0:
            continue
        hamming_distances = calc_hammingDist(query_hashes[iter, :], retrieval_hashes)
        cosine_distances = calc_euclideanDist(query_features[iter, :], retrieval_features)
        indexes_sorted = np.argsort(10*hamming_distances + cosine_distances)
        hamming_distances = hamming_distances[indexes_sorted]
        is_relevant_item = is_relevant_item[indexes_sorted]

        map["full"] += AP(is_relevant_item, len(is_relevant_item))

        for h in h_values:
            map[f"h={h}"] += AP(is_relevant_item, np.sum(hamming_distances<=h))

        for k in k_values:
            map[f"k={k}"] += AP(is_relevant_item, k)

    for key in map:
        map[key] = map[key] / num_query

    return map


def get_model_instance_results(model_instance_results_path, metrics=["mAP", "mAP_at_k=500", "mAP_at_h=0", "mAP_at_h=1", "mAP_at_h=2"]):
    hparams_to_ignore = [
        'commit',
        'no_skip',
        'datafolds',
        'semi_batch_ampliation',
        'num_workers'
    ]

    with open(model_instance_results_path,"r") as input_file:
        model_instance_results = json.load(input_file)
    
    model_instance_row = {}
    for metric in metrics:
        if metric in model_instance_results:
            model_instance_row[metric] = model_instance_results[metric]
        else:
            model_instance_row[metric] = -100

    model_instance_row['model_path'] = str(pathlib.Path(model_instance_results_path).parent).replace("eval","models")

    hparams = model_instance_results['hparams']
    for hparam_to_ignore in hparams_to_ignore:
        hparams.pop(hparam_to_ignore,None)
    
    model_instance_row.update(hparams)

    return model_instance_row

def get_results_dataframe(model_group, database, datafold, experiment_name=None, filtered_hparams={}, with_statistics=False, metrics=["mAP", "mAP_at_k=500", "mAP_at_h=0", "mAP_at_h=1", "mAP_at_h=2"]):

    EVAL_DIR = pathlib.Path("eval")

    results_file_suffix = "prediction_mAP"

    model_eval_path = EVAL_DIR / model_group / experiment_name / database

    results_df = None

    if model_eval_path.exists():

        index = 0

        for model_instance_eval_folder in model_eval_path.iterdir():

            model_instance_query_result_path = model_instance_eval_folder / f"query-{results_file_suffix}.json"
            model_instance_val_result_path = model_instance_eval_folder / f"val-{results_file_suffix}.json"

            if model_instance_query_result_path.exists() and model_instance_val_result_path.exists():

                model_instance_results_path = model_instance_eval_folder / f"{datafold}-{results_file_suffix}.json"
                model_instance_row = get_model_instance_results(model_instance_results_path,metrics)

                if with_statistics:
                    model_instance_statistics_path = model_instance_eval_folder / "statistics.json"
                    
                    if model_instance_statistics_path.exists():
                        with open(model_instance_statistics_path,"r") as input_file:
                            model_statistics = json.load(input_file)[datafold]
                    else:
                        model_statistics = {"centers":None,"deviations":None}
                    
                    model_instance_row.update(model_statistics)

                if results_df is not None:
                    results_df.loc[index] = model_instance_row
                else:
                    results_df = pd.DataFrame([model_instance_row])
            
                index += 1

    if experiment_name != 'None' and 'experiment_name' in results_df.columns:
        results_df = results_df[results_df['experiment_name']==experiment_name]
    if filtered_hparams is not None:
        for hparam in filtered_hparams:
            results_df = results_df[results_df[hparam].isin(filtered_hparams[hparam])]

    return results_df

def get_hparams_summary(df_row,hparams):
    
    list_summary = []

    for hparam in hparams:
        if isinstance(df_row[hparam],str):
            list_summary.append(f"{hparam}: '{df_row[hparam]}'")
        else:
            list_summary.append(f"{hparam}:{df_row[hparam]}")
        
    string_summary = ", ".join(list_summary)
    
    return string_summary

def get_summarized_dataframe(results_df, model_name, sort_by="mAP", metrics=["mAP", "mAP_at_k=500", "mAP_at_h=0", "mAP_at_h=1", "mAP_at_h=2"]):

    if results_df is not None and not results_df.empty:

        hparams = [col for col in results_df.columns if col not in metrics+["seed"]+['model_path']]

        summarized_results_df = results_df.groupby(by = hparams, as_index = False).agg({metric:['mean','std'] for metric in metrics})
        
        summarized_results_df['hparams_summary'] = flatten_df(summarized_results_df).apply(get_hparams_summary, args=[hparams], axis=1)

        summarized_results_df["model_class"] = model_name

        summarized_results_df = summarized_results_df[metrics+["model_class","hparams_summary"]+hparams]

        summarized_results_df = summarized_results_df.sort_values(by=(sort_by,'mean'), ascending=False)

        return summarized_results_df

    else:
        return None

def flatten_df(df):
    flat_df = df.copy()
    flat_df.columns = ["_".join(col).strip("_") for col in flat_df.columns.to_flat_index()]
    return flat_df

def get_metric_before_rotation(row,original_df,hparams_cols,metric):
    matching_row = original_df[(original_df[hparams_cols]==row[hparams_cols]).all(1)]
    return matching_row[metric].iloc[0]

def get_dataframe_before_after_rotation(model_group, database, datafold, experiment_name=None, filtered_hparams={}, with_statistics=True, metrics=["mAP", "mAP_at_k=500", "mAP_at_h=0", "mAP_at_h=1", "mAP_at_h=2"]):

    original_results_df = get_results_dataframe(model_group, database, datafold, experiment_name, filtered_hparams, with_statistics, metrics)
    results_df = get_results_dataframe("SONN", database, datafold, experiment_name, filtered_hparams, with_statistics, metrics)

    for metric in metrics:
        results_df[f"{metric}_before_rotation"] = -1

    for i, row in tqdm(results_df.iterrows()):
        for metric in metrics:
            results_df.at[i,f"{metric}_before_rotation"] = get_metric_before_rotation(row,original_results_df,hparams_cols[model_group],metric)

    return results_df

global hparams_cols

hparams_cols = {"QS":
                ["experiment_name",
                "database",
                "loss",
                "number_of_bits",
                "transformations",
                "architecture",
                "seed",
                "batch_size",
                "num_workers",
                "epochs",
                "patience",
                "learning_rate",
                "weight_decay",
                "optimizer",
                "penalty",
                "no_cube",
                "model"]
                }
