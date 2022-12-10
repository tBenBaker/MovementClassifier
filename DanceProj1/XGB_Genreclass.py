#imports
import pandas as pd
import numpy as np
import sys
sys.path.append("../../")
from DanceProj1.DanceObj import Dance
from DanceProj1.data_proc import get_data, data_to_features
import xgboost as xgb

def get_XGBclassifier():

    #load data
    dataBM, dataFM = get_data('../../aist_keypoints')    #get keypoint data
    dfBasic, dfAdvanced = data_to_features(dataBM, dataFM)  #get features as dataframes

    #make new indexes for Advanced, starting after last index in Basic, for unique index per id
    new_index_advanced = range(len(dfBasic.index), len(dfBasic.index)+len(dfAdvanced.index))
    dfAdvanced.index = dfAdvanced.index + new_index_advanced
    #alphabetize dfs by genre (to avoid reordering by classifiers later)
    dfBasic = dfBasic.sort_values(by='Genre')
    dfAdvanced = dfAdvanced.sort_values(by='Genre')

    from DanceProj1.data_proc import traintestval_split
    train, valid, testset = traintestval_split(dfBasic, dfAdvanced, 
                            testfrac_adv=.5, testfrac_bas=0, valfrac_adv_nonT=0, valfrac_bas=0)

    #split into X and y
    test_ids = testset['id']
    index_lookup = pd.DataFrame(
                testset.index, columns=['original_index']) #lookup table for original index

    y_test_strings = testset['Genre'].values
    X_test = testset.drop(['Genre'], axis=1).drop(['id'], axis=1)
    y_test = testset['Genre']
    X_valid = valid.drop(['Genre'], axis=1).drop(['id'], axis=1)
    y_valid = valid['Genre']
    X_train = train.drop(['Genre'], axis=1).drop(['id'], axis=1)
    y_train = train['Genre']

    #make genre labels numeric
    labels = y_test.unique()

    y_trainxgb = y_train.replace(labels, list(range(len(labels))), regex=True)
    #y_validxgb = y_valid.replace(labels, list(range(len(labels))), regex=True)
    y_test = y_test.replace(labels, list(range(len(labels))), regex=True)
    X_test = X_test.replace(labels, list(range(len(labels))), regex=True)

    xgb_mod = xgb.XGBClassifier(objective='multi:softprob', eta =.2, max_depth =8, subsample =.6, random_state=42, n_jobs=16) 
    #learning rate, max depth and subsample tuned empirically
    xgb_mod.fit(X_train, y_trainxgb)
    y_pred = xgb_mod.predict(X_test)
    y_proba_pred = xgb_mod.predict_proba(X_test)
    yprobdf = pd.DataFrame(y_proba_pred, columns=labels)
    yprobdf['Original Index'] = index_lookup['original_index']
    yprobdf['True_Label'] = y_test_strings
    yprobdf['id'] = test_ids.values

    return xgb_mod, X_test, y_test, y_pred, y_proba_pred, yprobdf
    