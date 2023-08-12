import os
import time

from NeuralNet import NNModel
from definitions import TaskType
from preprocssing import load_data, generate_data

import numpy as np

def main():
    #If True will generate data and save to 'data' folder.
    should_generate_data = True
    run_voice_2face_arbitration = True
    run_voice_face_matching = True
    run_benchmark_classifiers = False
    scale_data = False
    only_load_and_predict = False
    train_perc = 0.7 #fraction of data for training
    validation_perc = 0.2 #fraction of data for validation
    # Fraction for test set is simply what's left

    this_file_path = os.path.dirname(os.path.abspath(__file__)) + '/'

    if run_benchmark_classifiers:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.svm import SVC

    """
    VOICE-FACE/FACE ARBITRATION
    """
    if run_voice_2face_arbitration:
        #load/generate data
        if should_generate_data:
            img_vecs, img_labels, aud_vecs, aud_labels, num_labels = load_data()

            (X_train, y_train,
             X_val, y_val,
             X_test, y_test) = generate_data(img_vecs, img_labels, aud_vecs, aud_labels, num_labels,
                                             perc_train=train_perc, per_valid=validation_perc,
                                             task_type=TaskType.VFF_ARBITRATION,
                                             scale_data=scale_data)

            np.save(this_file_path + 'data/VFF_X_train.npy', X_train)
            np.save(this_file_path + 'data/VFF_y_train.npy', y_train)
            np.save(this_file_path + 'data/VFF_X_val.npy', X_val)
            np.save(this_file_path + 'data/VFF_y_val.npy', y_val)
            np.save(this_file_path + 'data/VFF_X_test.npy', X_test)
            np.save(this_file_path + 'data/VFF_y_test.npy', y_test)

        else:
            X_train = np.load(this_file_path +'data/VFF_X_train.npy')
            y_train = np.load(this_file_path +'data/VFF_y_train.npy')
            X_val =   np.load(this_file_path +'data/VFF_X_val.npy')
            y_val =   np.load(this_file_path +'data/VFF_y_val.npy')
            X_test =  np.load(this_file_path +'data/VFF_X_test.npy')
            y_test =  np.load(this_file_path +'data/VFF_y_test.npy')

        print('Voice-Face/Face Arbitration:')
        print('----------------------------')


        #Try a few out-of-the-box classifiers as benchmarks
        if run_benchmark_classifiers:
            classifiers = {
                'logistic regression' : LogisticRegression(C=0.08),
                'decision tree' : DecisionTreeClassifier(min_samples_leaf=10),
                'Boosting Trees' :  HistGradientBoostingClassifier(),
                # 'SVM' : SVC() #takes too long
            }

            for name, clf in classifiers.items():
                t0 = time.time()
                clf.fit(X_train, y_train)
                print(f'Using {name} took {time.time() - t0} seconds:')
                print(f'test accuracy = {clf.score(X_test, y_test)}')
                print(f'validation accuracy = {clf.score(X_val, y_val)}')
                print(f'train accuracy = {clf.score(X_train, y_train)}')
                print()



        # Train network and display accuracy results for all data sets
        t0 = time.time()
        trainer = NNModel(task_type=TaskType.VFF_ARBITRATION)
        if only_load_and_predict:
            trainer.load(dir_path='models', name='VFF')
        else:
            trainer.train(X_train,y_train, X_val, y_val,
                          lr=5e-3, batch_size=64, num_epochs=42)
            trainer.save(dir_path='models', name='VFF')
        print(f'Using NN took {time.time() - t0} seconds:')
        print(f'test accuracy = {trainer.score(X_test, y_test)}')
        print(f'validation accuracy = {trainer.score(X_val, y_val)}')
        print(f'train accuracy = {trainer.score(X_train, y_train)}')

    """
    VOICE-FACE MATCHING
    """

    if run_voice_face_matching:
        #load/generate data
        if should_generate_data:
            img_vecs, img_labels, aud_vecs, aud_labels, num_labels = load_data()

            (X_train, y_train,
             X_val, y_val,
             X_test, y_test) = generate_data(img_vecs, img_labels, aud_vecs, aud_labels, num_labels,
                                             perc_train=train_perc, per_valid=validation_perc,
                                             task_type=TaskType.VF_MATCHING,
                                             scale_data=scale_data)

            np.save(this_file_path +'data/Matchdata_X_train.npy',X_train)
            np.save(this_file_path +'data/Matchdata_y_train.npy',y_train)
            np.save(this_file_path +'data/Matchdata_X_val.npy',X_val)
            np.save(this_file_path +'data/Matchdata_y_val.npy',y_val)
            np.save(this_file_path +'data/Matchdata_X_test.npy',X_test)
            np.save(this_file_path +'data/Matchdata_y_test.npy',y_test)

        else:
            X_train = np.load(this_file_path +'data/Matchdata_X_train.npy')
            y_train = np.load(this_file_path +'data/Matchdata_y_train.npy')
            X_val =   np.load(this_file_path +'data/Matchdata_X_val.npy')
            y_val =   np.load(this_file_path +'data/Matchdata_y_val.npy')
            X_test =  np.load(this_file_path +'data/Matchdata_X_test.npy')
            y_test =  np.load(this_file_path +'data/Matchdata_y_test.npy')

        print('Voice-Face Matching:')
        print('--------------------')

        #Try a few out-of-the-box classifiers as benchmarks
        if run_benchmark_classifiers:
            classifiers = {
                'logistic regression' : LogisticRegression(C=0.05),
                'decision tree' : DecisionTreeClassifier(min_samples_leaf=10),
                'Boosting Trees' :  HistGradientBoostingClassifier(),
                # 'SVM' : SVC() #takes too long
            }

            for name, clf in classifiers.items():
                t0 = time.time()
                clf.fit(X_train, y_train)
                print(f'Using {name} took {time.time() - t0} seconds:')
                print(f'test accuracy = {clf.score(X_test, y_test)}')
                print(f'validation accuracy = {clf.score(X_val, y_val)}')
                print(f'train accuracy = {clf.score(X_train, y_train)}')
                print()



        # Train network and display accuracy results for all data sets
        t0 = time.time()
        trainer = NNModel(task_type=TaskType.VF_MATCHING)
        if only_load_and_predict:
            trainer.load(dir_path='models', name='Matching')
        else:
            trainer.train(X_train,y_train, X_val, y_val,
                          lr=5e-3, batch_size=64, num_epochs=17)
            trainer.save(dir_path='models', name='Matching')
        print(f'Using NN took {time.time() - t0} seconds:')
        print(f'test accuracy = {trainer.score(X_test, y_test)}')
        print(f'validation accuracy = {trainer.score(X_val, y_val)}')
        print(f'train accuracy = {trainer.score(X_train, y_train)}')


    print('done')


if __name__ == '__main__':
    main()