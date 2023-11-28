import numpy as np

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - B1 @ B2.T)
    return distH


def calc_euclideanDist(F1, F2):
    distE = np.sum( (F1-F2)**2, axis = 1)
    return distE


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