import pickle
from pickle import HIGHEST_PROTOCOL
from typing import Dict
from itertools import product

from definitions import TaskType

import numpy as np


def load_data():

    #Load pickle files
    img_feat_dic, aud_feat_dic = load_files()

    #extract persons names and the embedding vector
    img_names, img_vecs = extract_name_vec(img_feat_dic)
    aud_names, aud_vecs = extract_name_vec(aud_feat_dic)

    #Keep only the names that appear both in images and audios.
    #Replace the name with an enumerated value
    img_vecs, img_labels, aud_vecs, aud_labels, num_labels = intersect_and_label(img_names, img_vecs, aud_names, aud_vecs)

    return  (np.array(img_vecs, dtype=np.float64), np.array(img_labels),
             np.array(aud_vecs, dtype=np.float64), np.array(aud_labels), num_labels)


def load_files():
    """
    Loads the embeddings from the pickle files
    """
    with open('data/image_embeddings.pickle', 'rb') as f:
        img_feat_dic = pickle.load(f)

    with open('data/audio_embeddings.pickle', 'rb') as f:
        aud_feat_dic = pickle.load(f)

    return img_feat_dic, aud_feat_dic


def extract_name_vec(feat_dict: Dict[str, np.ndarray]):
    """
    Extract the person name and embedding vector for each sample
    """

    names = []
    vecs = []

    for file_name, vec in feat_dict.items():
        name = file_name.split('/')[0]
        names.append(name)
        vecs.append(vec)

    return names, vecs

def intersect_and_label(img_names, img_vecs, aud_names, aud_vecs):
    """
    Keep only the names that appear both in images and audios.
    Replace the name with an enumerated value
    """

    #Find the unique set of names
    unique_names_set = set(img_names).intersection(set(aud_names))
    #enumerate them
    name_label_dict = dict(zip(sorted(list(unique_names_set)), range(len(unique_names_set))))

    def _filter(_names, _vecs):
        _vecs_cor = []
        labels = []
        for name, vec in zip(_names, _vecs):
            if name in name_label_dict:
                _vecs_cor.append(vec)
                labels.append(name_label_dict[name])
        return _vecs_cor, labels

    #filter out labels that are not in the intersection
    img_vecs, img_labels = _filter(img_names, img_vecs)
    aud_vecs, aud_labels = _filter(aud_names, aud_vecs)

    return img_vecs, img_labels, aud_vecs, aud_labels, len(unique_names_set)



def generate_data(img_vecs, img_labels, aud_vecs, aud_labels, num_labels,
                  perc_train, per_valid, task_type, scale_data=False):
    """
    Prepares the data to be ready for classification
    """
    all_labels = np.arange(num_labels)
    np.random.shuffle(all_labels)

    # Split to train/validation/test sets
    # The splitting is done on the labels so there's no "leakage" between datasets
    N_train = int(perc_train * num_labels)
    N_val = int(per_valid * num_labels)
    train_labels =  all_labels[:N_train]
    val_labels =  all_labels[N_train:N_train+N_val]
    test_labels =  all_labels[N_train+N_val:]

    if task_type==TaskType.VF_MATCHING:
        generator = _generate_matching_data_for_labels
    elif task_type==TaskType.VFF_ARBITRATION:
        generator = _generate_VFF_data_for_labels
    else:
        raise Exception('...')


    def _gen_for_subset(sub_labels):
        #find the indices of data points the belong to subset of labels
        sub_img_inds = np.isin(img_labels, sub_labels)
        sub_aud_inds = np.isin(aud_labels, sub_labels)

        #generate data for this subset
        X_sub, y_sub = generator(img_vecs[sub_img_inds], img_labels[sub_img_inds],
                                 aud_vecs[sub_aud_inds], aud_labels[sub_aud_inds],
                                 sub_labels)
        return np.array(X_sub), np.array(y_sub)

    X_train, y_train = _gen_for_subset(train_labels)
    X_val, y_val = _gen_for_subset(val_labels)
    X_test, y_test = _gen_for_subset(test_labels)


    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)

    if scale_data:
        # Shift training data to zero mean and scale it (z-scaling)
        # validation and test set are using the training data mean and std
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std
        X_test = (X_test - x_mean) / x_std

    return (X_train, y_train,
            X_val, y_val,
            X_test, y_test)



def _generate_matching_data_for_labels(img_vecs, img_labels, aud_vecs, aud_labels, labels, concat=True):
    """
    Generate data for voice-face matching
    """
    y = []
    X = []
    for label in labels:
        #for each label find the relevant data subset for that label
        img_inds = img_labels == label
        rel_imgs = img_vecs[img_inds]
        aud_inds = aud_labels == label
        rel_auds = aud_vecs[aud_inds]

        #each possible voice-face pair for that label are added as a data point
        matching_pairs = product(rel_imgs, rel_auds)
        if concat:
            matching_vecs = [np.concatenate(pair) for pair in matching_pairs]
        else:
            matching_vecs = list(matching_pairs)
        y += [1] * len(matching_vecs) #the classification label is 1 for 'match'
        X += matching_vecs

        # for each matching pair create a non-matching pair (so to keep the dataset balanced)
        # a random subset of non-matching audio vectors is used
        neg_inds_sample = np.random.choice(np.where(np.logical_not(aud_inds))[0], size=len(rel_auds))
        neg_auds = aud_vecs[neg_inds_sample]
        non_matching_pairs = product(rel_imgs, neg_auds)
        if concat:
            non_matching_vecs = [np.concatenate(pair) for pair in non_matching_pairs]
        else:
            non_matching_vecs = list(non_matching_pairs)

        y += [0] * len(non_matching_vecs)
        X += non_matching_vecs

    return X, y



def _generate_VFF_data_for_labels(img_vecs, img_labels, aud_vecs, aud_labels, labels, concat=True):
    """
    Generate data for voice-face-face arbitration
    """

    y = []
    X = []
    for label in labels:
        # for each label find the relevant data subset for that label
        aud_inds = aud_labels == label
        rel_auds = aud_vecs[aud_inds]

        pos_img_pos = img_labels == label
        pos_img_inds = np.where(pos_img_pos)[0] #indices of image vectors with given label
        neg_img_inds = np.where(np.logical_not(pos_img_pos))[0]#indices of image vectors with OTHER label
        for aud_vec in rel_auds:
            #for each audio vector with given label select one matching and one non-matching image vector (to keep dataset balanced)
            pos_img_sample = img_vecs[np.random.choice(pos_img_inds)]
            neg_img_sample = img_vecs[np.random.choice(neg_img_inds)]
            if np.random.choice(2):#the position of the matching image in the concatenation should also be random
                x_row = (aud_vec, pos_img_sample, neg_img_sample)
                y.append(1)
            else:
                x_row = (aud_vec, neg_img_sample, pos_img_sample)
                y.append(0)

            x_row = np.concatenate(x_row) if concat else x_row
            X.append(x_row)

    return X, y


def _create_triplets():
    img_vecs, img_labels, aud_vecs, aud_labels, num_labels = load_data()
    labels = np.unique(aud_labels)
    X, y = _generate_VFF_data_for_labels(img_vecs, img_labels, aud_vecs, aud_labels, labels, concat=False)

    with open('data/triplets.pickle', 'wb') as f:
        pickle.dump(X, f, protocol=HIGHEST_PROTOCOL)

    with open('data/triplets_y.pickle', 'wb') as f:
        pickle.dump(y, f, protocol=HIGHEST_PROTOCOL)


def _create_pairs():
    img_vecs, img_labels, aud_vecs, aud_labels, num_labels = load_data()
    labels = np.unique(aud_labels)
    X, y = _generate_matching_data_for_labels(img_vecs, img_labels, aud_vecs, aud_labels, labels, concat=False)

    with open('data/triplets.pickle', 'wb') as f:
        pickle.dump(X, f, protocol=HIGHEST_PROTOCOL)

    with open('data/triplets_y.pickle', 'wb') as f:
        pickle.dump(y, f, protocol=HIGHEST_PROTOCOL)


def my_test():
    img_vecs, img_labels, aud_vecs, aud_labels, num_labels = load_data()

    # aud_vecs = aud_vecs / np.linalg.norm(aud_vecs, axis=1)[:, np.newaxis]
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test) = generate_data(img_vecs, img_labels, aud_vecs, aud_labels, num_labels,
                                     perc_train=0.7, per_valid=0.2, task_type=TaskType.VFF_ARBITRATION)

    # np.save('data/Matchdata_X_train.npy',X_train)
    # np.save('data/Matchdata_y_train.npy',y_train)
    # np.save('data/Matchdata_X_val.npy',X_val)
    # np.save('data/Matchdata_y_val.npy',y_val)
    # np.save('data/Matchdata_X_test.npy',X_test)
    # np.save('data/Matchdata_y_test.npy',y_test)

    np.save('data/VFF_X_train.npy',X_train)
    np.save('data/VFF_y_train.npy',y_train)
    np.save('data/VFF_X_val.npy',X_val)
    np.save('data/VFF_y_val.npy',y_val)
    np.save('data/VFF_X_test.npy',X_test)
    np.save('data/VFF_y_test.npy',y_test)



if __name__ == '__main__':
    my_test()

