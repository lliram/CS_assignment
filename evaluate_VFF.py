import os
import pickle

import numpy as np
from typing import List, Tuple, Union

from NeuralNet import NNModel
from definitions import TaskType


def evaluate_accuracy_for_embeddings(vff_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                     y: Union[np.array, List]):
    """
    Evaluates the accuracy of the model on some input set.

    :param vff_triplets: A tuple of the embedding vectors (numpy arrays) in the order (audio, image, image)
    :param y: list/numpy array of values 0/1 that indicate which image matches the voice. 0 for the left (2nd in tuple)
    and 1 for the right (3rd in the tuple).
    :return: The average accuracy
    """

    y_pred = predict_for_embeddings(vff_triplets)
    acc = (y_pred == y).mean()
    print(f'Accuracy = {acc}')
    return acc

def predict_for_embeddings(vff_triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """
    Prediction using the model
    :param vff_triplets: A tuple of the embedding vectors (numpy arrays) in the order (audio, image, image)
    :return: Model prediction: numpy array of values 0/1 that indicate which image matches the voice. 0 for the left (2nd in tuple)
    and 1 for the right (3rd in the tuple).
    """
    nn_model = NNModel(task_type=TaskType.VFF_ARBITRATION)
    path = os.path.dirname(os.path.abspath(__file__)) + '/models'
    nn_model.load(dir_path=path, name='VFF')

    X = [np.concatenate(triplet) for triplet in vff_triplets]
    X = np.array(X)
    y_pred = nn_model.predict(X)

    return y_pred


def load_triplets_evaluate(triplet_file_name, y_file_name):
    """
    Loads triplets from pickle file and runs 'evaluate_accuracy_for_embeddings'
    :param triplet_file_name:
    :param y_file_name:
    :return:
    """
    with open(triplet_file_name, 'rb') as f:
        triplets = pickle.load(f)

    with open(y_file_name, 'rb') as f:
        y = pickle.load(f)

    evaluate_accuracy_for_embeddings(triplets, y)

def main():
    from preprocssing import create_triplets
    create_triplets() #provide path to pickle files (otherwise will use provided input data)
    path = os.path.dirname(os.path.abspath(__file__))  + '/'
    load_triplets_evaluate(path + 'data/triplets.pickle', path + 'data/triplets_y.pickle')



if __name__ == '__main__':
    main()