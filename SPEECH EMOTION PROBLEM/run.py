import librosa
from fastai.vision import *
import librosa.display

import pandas as pd
import re
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pathlib import Path

"""Function to extract features from sound"""

def feature_extractor3(folder_path):
    df = pd.DataFrame(columns=['feature', 'label'])
    audio_files_dir = Path(folder_path)
    bookmark=0
    duration = 10
    file_name_list = []
    for label in ['neutral','happy','disgust','sad','fear']:
        label_str = label
        audio_files = list(Path(audio_files_dir/label).glob('*.wav'))

        print('Extracting ' + folder_path + label)

        for audio_file in audio_files:
            file_name_list.append(audio_file.as_posix().split('/')[-1])

            samples, sr = librosa.load(audio_file, res_type='kaiser_fast',duration=duration,sr=16000*2)
            # print(samples.shape, sample_rate)
            samples = np.concatenate([samples, np.zeros(duration*sr-samples.shape[0])])
            # print(samples.shape)

            mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=50, hop_length=256)

            # print(mfcc.shape)
            # print(spectral_center.shape)
            # print(chroma.shape)
            # print(spectral_contrast.shape)
            # print(spectral_bandwidth.shape)
            # print(flatness.shape)
            # print(zero_crossing.shape)
            # print(fourier_tempogram.shape, "\n---\n")

            # data = np.concatenate([mfcc, spectral_center, chroma, spectral_contrast, spectral_bandwidth, flatness, zero_crossing], axis = 0)
            data = np.concatenate([mfcc], axis = 0)

            df.loc[bookmark] = [data, label_str]
            bookmark=bookmark+1

    return df

def get_file_names(folder_name):
    audio_files_dir = Path(folder_name)
    file_name_list = []
    for label in ['neutral','happy','disgust','sad','fear']:
        audio_files = list(Path(audio_files_dir/label).glob('*.wav'))
        for audio_file in audio_files:
            file_name_list.append(audio_file.as_posix().split('/')[-1])

    return file_name_list

def extract_features(folder_path, feature_type):
    df = feature_extractor3(folder_path)
    df.to_pickle(folder_path+"features_7_"+feature_type+".csv")

def main(folder_path):

    print("Extracting features(this might take some time)...")
    extract_features(folder_path, 'test')

    df_train = pd.read_pickle('meld/features_5_train.csv')
    df_test = pd.read_pickle(folder_path+"features_7_test.csv")

    # print(df_test)

    x_train1 = np.array(df_train.feature.values.tolist())
    x_train = np.mean(x_train1, axis=1)
    y_train = np.array(df_train.label.tolist())

    x_test1 = np.array(df_test.feature.values.tolist())
    x_test = np.mean(x_test1, axis=1)
    y_test = np.array(df_test.label.tolist())

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    print('fitting data....')
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))

    clf.fit(x_train, y_train)

    print('Classifying(this might take some time).............')
    y_pred=clf.predict(x_train)
    print("Train accuracy    :",accuracy_score(y_true=y_train, y_pred=y_pred))

    y_pred=clf.predict(x_test)
    print("Validation accuracy: ",accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Weighted FScore: ', precision_recall_fscore_support(y_test, y_pred, average='weighted'))


    test_file_names = get_file_names(folder_path)

    submission = pd.DataFrame(columns=['File name', 'prediction'])
    submission['File name'] = test_file_names
    submission['prediction'] = y_pred

    submission.to_csv('output.csv', index = False)

    print("Finished!")


# incase you need to train your features again then uncomment this
# extract_features(folder_path, 'train')
main(str(sys.argv[1]))
