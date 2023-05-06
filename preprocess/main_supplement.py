#from create_dataset import create_MIMIC_dataset, create_eICU_dataset
#from dataframe_gen import preprocess
#from numpy_convert import convert2numpy
#from preprocess_utils import label_npy_file
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import stratifiers_utils
from stratifiers_utils import MultilabelStratifiedKFold
#from scipy.sparse import spmatrix

from numpy_convert import convert2numpy
#  from pandas.compat import pickle
#import pickle

# load pickled DataFrame using pandas.compat.pickle

def multi_hot(column):
    mlb = MultiLabelBinarizer()
    encoded_dx = mlb.fit_transform(column)
    multihot_vector=mlb.classes_
    print(multihot_vector)
    df_dx = pd.DataFrame(encoded_dx, columns=multihot_vector)
    df_dx.columns = df_dx.columns.astype(np.int16)
    df_dx = df_dx.sort_index(axis=1)
    return df_dx


def label_npy_file(input_path, output_path):
    columns_lst = ['readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis']
    #columns_lst = ['readmission', 'mortality', 'los_3day', 'los_7day']
    # path need to exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'label'), exist_ok=True)
    for src in ['mimic', 'eicu', 'pooled']:
        filename = '{}_df.pkl'.format(src)
        df = pd.read_pickle(os.path.join(input_path, filename))
        
        for col in columns_lst: 
            column = df[col]
            if col =='diagnosis':
                column = multi_hot(column)
                if column.shape[1] == 17:
                    zeros = pd.Series(list(np.zeros(column.shape[0], dtype='int16')))
                    column[15]= zeros
                    column = column.reindex(sorted(column.columns), axis=1)
            array = np.array(column, dtype='int16')
            save_name = f'{src}_{col}.npy'
            print('save path', output_path)
            print('save savename', save_name)
            np.save(os.path.join(output_path, 'label', save_name), array)
            print(f'{src}_{col}.npy',  'numpy save finish')


def train_valid_test_split(target_cols, df, random_state, test_ratio, train_valid_fold):
#def train_valid_test_split(target_cols, df, random_state, test_ratio): # without dx
    sss_train_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state =random_state)
    #mskf = MultilabelStratifiedKFold(n_splits=train_valid_fold, random_state=random_state)
    mskf = MultilabelStratifiedKFold(n_splits=train_valid_fold)
    
    X = df.copy()
    for target in target_cols:
        #if target == 'dx':
        if target == 'diagnosis':
            continue
        print("____{}____ train and test split start!".format(target))
        fold_column = '{}_fold'.format(target)
        X[fold_column]=1
        print('X fold_column \n',X[fold_column].value_counts())
        y = X[target]

        #train / test split 4:1
        for train_index, test_index in sss_train_test.split(X,y):
            X_test = X.loc[test_index]
            X_test.loc[test_index, fold_column]=0    

        #train = -1 , test = 0
        X_train = X.loc[train_index].reset_index(drop=True)
        y_train = y.loc[train_index].reset_index(drop=True)

        #train / valid slpit 4:1 
        for train_index, valid_index in sss_train_test.split(X_train,y_train):
            X_valid = X_train.loc[valid_index]
            X_valid.loc[valid_index, fold_column]=2


        #test = 0 train = 1 valid = 2
        X_train = X_train.loc[train_index].reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        X = pd.concat([X_train, X_valid, X_test], axis=0).reset_index(drop=True)

        print('fold_split_results!!!!! \n', X[fold_column].value_counts())
        
        #return X 

        
        #Multi_label_stratified_Kfold
         
    #diagnosis multi_label stratified_Kfold
    print('___diagnosis multi_label_stratified_split____') 
    X['diagnosis_fold'] = 1
    y = multi_hot(X['diagnosis'])
    #y = X[columns]
    for i, (train_index, test_index) in enumerate(mskf.split(X,y)):
        if i != 0 :
            continue
        elif i== 0:
            trn_index = train_index
            X_test=X.loc[test_index]
            X_test.loc[test_index, 'diagnosis_fold'] = 0

    #X_train 
    X_train = X.loc[trn_index].reset_index(drop=True)
    y_train = y.loc[trn_index].reset_index(drop=True)
    for i, (train_index, valid_index) in enumerate(mskf.split(X_train, y_train)):
        if i != 0 :
            continue
        elif i== 0:
            trn_index = train_index
            X_valid= X_train.loc[valid_index]
            X_valid.loc[valid_index, 'diagnosis_fold'] = 2

    #test = 0 train = 1 valid = 2
    X_train = X_train.loc[trn_index].reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X = pd.concat([X_train, X_valid,X_test]).reset_index(drop=True)
    print('fold_split_results!!!!! \n', X['diagnosis_fold'].value_counts())
    
    
    return X        



def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_input_path', type=str)
    parser.add_argument('--data_output_path', type=str)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--test_ratio', type=float, default = 0.4)
    parser.add_argument('--train_valid_fold', type=int, default = 10)
    return parser

def main():
    args = get_parser().parse_args()
    # file names
     
    #convert2numpy(args.data_input_path, args.data_output_path)
    label_npy_file(args.data_input_path, args.data_output_path)
    os.makedirs(os.path.join(args.data_output_path, 'label'), exist_ok=True)
    
        
    target_cols = ['readmission', 'mortality', 'los_3day', 'los_7day', 'diagnosis']  

    for src in ['mimic', 'eicu', 'pooled']: 
        print(f'{src} data: ')
        filename = '{}_df.pkl'.format(src)
        df = pd.read_pickle(os.path.join(args.data_input_path, filename))

        X = train_valid_test_split(target_cols, df, args.seed, args.test_ratio, args.train_valid_fold)
        #X = train_valid_test_split(target_cols, df, args.seed, args.test_ratio)
        
        save_name = f'{src}_{args.seed}_fold_split.csv'
        #print('save path', args.data_output_path)
        #print('save savename', save_name)
        
        X.to_csv(os.path.join(args.data_output_path, 'fold', save_name), index=False)
          
    
    print('preprocess finish!!')        


if __name__ == '__main__':
    main()





