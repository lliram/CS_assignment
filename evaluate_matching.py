import os
import pickle

import numpy as np
from typing import List, Tuple, Union

from NeuralNet import NNModel
from definitions import TaskType


def evaluate_accuracy_for_embeddings(vf_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                     y: Union[np.array, List]):
    """
    Evaluates the accuracy of the model on some input set.

    :param vf_pairs: A tuple of the embedding vectors (numpy arrays) in the order (image, audio)
    :param y: list/numpy array of values 0/1 that indicate whether the audio and image match.
    0 -not matching, 1 - matching
    :return: The average accuracy
    """

    y_pred = predict_for_embeddings(vf_pairs)
    acc = (y_pred == y).mean()
    print(f'Accuracy = {acc}')
    return acc

def predict_for_embeddings(vf_pairs: List[Tuple[np.ndarray, np.ndarray]]):
    """
    Prediction using the model
    :param vf_pairs: A pair of the embedding vectors (numpy arrays) in the order (image, audio)
    :return: Model prediction: numpy array of values 0/1 that indicate whether the audio and image match.
    0 -not matching, 1 - matching
    """

    nn_model = NNModel(task_type=TaskType.VF_MATCHING)
    path = os.path.dirname(os.path.abspath(__file__)) + '/models'
    nn_model.load(dir_path=path, name='Matching')

    X = [np.concatenate(pair) for pair in vf_pairs]
    X = np.array(X)
    y_pred = nn_model.predict(X)

    return y_pred


def load_pairs_evaluate(pairs_file_name, y_file_name):
    """
    Loads pairs from pickle file and runs 'evaluate_accuracy_for_embeddings'
    :param pairs_file_name:
    :param y_file_name:
    :return:
    """
    with open(pairs_file_name, 'rb') as f:
        pairs = pickle.load(f)

    with open(y_file_name, 'rb') as f:
        y = pickle.load(f)

    evaluate_accuracy_for_embeddings(pairs, y)

def _test():
    from preprocssing import _create_pairs
    _create_pairs()
    path = os.path.dirname(os.path.abspath(__file__))  + '/'
    load_pairs_evaluate(path + 'data/triplets.pickle', path + 'data/triplets_y.pickle')



if __name__ == '__main__':
    _test()