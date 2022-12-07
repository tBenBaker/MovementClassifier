
import pickle
import glob
import numpy as np
import pandas as pd
import sys
sys.path.append("../../")

from DanceProj1.DanceObj import Dance


def get_data(path):
    #allfilenames = glob.glob(f'{path}/*')         
    #get lists of filenames per genre, basic and advanced
    breakfilesBM = glob.glob(f'{path}/gBR_sBM*')
    breakfilesFM = glob.glob(f'{path}/gBR_sFM*')
    popfilesBM = glob.glob(f'{path}/gPO_sBM*')
    popfilesFM = glob.glob(f'{path}/gPO_sFM*')
    lockfilesBM = glob.glob(f'{path}/gLO_sBM*')
    lockfilesFM = glob.glob(f'{path}/gLO_sFM*')
    midhopfilesBM = glob.glob(f'{path}/gMH_sBM*')
    midhopfilesFM = glob.glob(f'{path}/gMH_sFM*')
    lahopfilesBM = glob.glob(f'{path}/gLH_sBM*')
    lahopfilesFM = glob.glob(f'{path}/gLH_sFM*')
    housefilesBM = glob.glob(f'{path}/gHO_sBM*')
    housefilesFM = glob.glob(f'{path}/gHO_sFM*')
    waackfilesBM = glob.glob(f'{path}/gWA_sBM*')
    waackfilesFM = glob.glob(f'{path}/gWA_sFM*')
    krumpfilesBM = glob.glob(f'{path}/gKR_sBM*')
    krumpfilesFM = glob.glob(f'{path}/gKR_sFM*')
    sjazzfilesBM = glob.glob(f'{path}/gJS_sBM*')
    sjazzfilesFM = glob.glob(f'{path}/gJS_sFM*')
    bjazzfilesBM = glob.glob(f'{path}/gJB_sBM*')
    bjazzfilesFM = glob.glob(f'{path}/gJB_sFM*')

    #make list of lists of filesnames by basic/advanced
    genrefilesBM = [breakfilesBM, popfilesBM, lockfilesBM, midhopfilesBM, lahopfilesBM,
                    housefilesBM, waackfilesBM, krumpfilesBM, sjazzfilesBM, bjazzfilesBM]
    genrefilesFM = [breakfilesFM, popfilesFM, lockfilesFM, midhopfilesFM, lahopfilesFM, 
                    housefilesFM, waackfilesFM, krumpfilesFM, sjazzfilesFM, bjazzfilesFM]
    
    #initialize dictionaries for position arrays, per basdic/advanced per genre
    genredataBM = {'Break':[], 'Pop':[], 'Lock':[], 'Midhop':[], 'LAhop':[], 'House':[], 'Waack':[],
                    'Krump':[], 'Street Jazz':[], 'Ballet Jazz':[]}
    genredataFM = {'Break':[], 'Pop':[], 'Lock':[], 'Midhop':[], 'LAhop':[], 'House':[], 'Waack':[],
                    'Krump':[], 'Street Jazz':[], 'Ballet Jazz':[]}


    for i, genre in enumerate(genrefilesBM):
        for filename in genre:
            unpickled = pickle.load(open(filename, 'rb'))['keypoints3d_optim'] #unpickle the (optim) position arrays
            id = filename.split('/')[-1].split('.')[0]                        #get the id from the filename
            pos = np.swapaxes(unpickled,0,1)                                   #swap axes to get (joints, frames xyz)
            pos = np.delete(pos, [2,3], 0)                                     #delete eyes
            genredataBM[list(genredataBM.keys())[i]].append((pos,id))          #add to dictionary

    for i, genre in enumerate(genrefilesFM):                                   #repeat for advanced dances
        for filename in genre:
            unpickled = pickle.load(open(filename, 'rb'))['keypoints3d_optim']
            id = filename.split('/')[-1].split('.')[0]
            pos = np.swapaxes(unpickled,0,1)
            pos = np.delete(pos, [2,3], 0)
            genredataFM[list(genredataFM.keys())[i]].append((pos,id))

    return genredataBM, genredataFM 
    
def data_to_features(dataBM, dataFM):

    featuresBM = []                     # list for feature-dictionaries for all basic dances                          
    for genre in dataBM:
        for i in range(len(dataBM[genre])):
            dance = Dance(dataBM[genre][i][0], 1/60)   #create Dance object for each basic dance
            dance.genre = genre                             #add genre
            dance.id = dataBM[genre][i][1]             #add id
            dance.get_features()                            #get features
            featuresBM.append(dance.features)                   #add feature-dict to list
    
    featuresFM = []                     # repeat for all advanced dances
    for genre in dataFM:
        for i in range(len(dataFM[genre])):
            dance = Dance(dataFM[genre][i][0], 1/60)
            dance.genre = genre
            dance.id = dataFM[genre][i][1]
            dance.get_features()
            featuresFM.append(dance.features)
            
    #turn these into dataframes
    dfBM = pd.DataFrame(featuresBM)
    dfFM = pd.DataFrame(featuresFM)            
            
    return dfBM, dfFM
        
     
def traintestval_split(dfBasic, dfAdvanced, testfrac_adv=.5, testfrac_bas=0, valfrac_adv_nonT=.2, valfrac_bas=.12):

    testset = pd.DataFrame(columns=dfAdvanced.columns)  #initialize testset
    Genres = list(dfAdvanced.Genre.unique())        #get list of genres
    for genre in Genres:                        #for each genre
        test_adv = dfAdvanced.loc[dfAdvanced.Genre == genre].sample(frac=testfrac_adv, random_state=1) #get 50% of advanced dances
        testset = pd.concat([testset, test_adv])        #add to testset

    nontest_advanced = dfAdvanced.drop(testset.index)   #get the rest of the advanced dances

    test_bas = pd.DataFrame(columns=dfBasic.columns)    #initialize testset for basic dances
    for genre in Genres:
        test_bas = dfBasic.loc[dfBasic.Genre == genre].sample(frac=testfrac_bas, random_state=1)
        testset = pd.concat([testset, test_bas])        #add to testset

    valid_adv = pd.DataFrame(columns=dfAdvanced.columns)    #same for validation set
    for genre in Genres:
        val = nontest_advanced.loc[nontest_advanced.Genre == genre].sample(frac=valfrac_adv_nonT, random_state=1)
        valid_adv = pd.concat([valid_adv, val])

    train_adv = nontest_advanced.drop(valid_adv.index)    #nontest, nonval advanced dances are training set

    valid_bas = pd.DataFrame(columns=dfBasic.columns)   #but validation pulls from basic dances too
    Genres = list(dfAdvanced.Genre.unique())
    for genre in Genres:
        val = dfBasic.loc[dfBasic.Genre == genre].sample(frac=valfrac_bas, random_state=1)
        valid_bas = pd.concat([valid_bas, val])

    train_bas = dfBasic.drop(valid_bas.index)   #training set includes the nonval basic dances
    train_bas = train_bas.drop(test_bas.index)  #and the non-test basic dances (none, by default)

    train = pd.concat([train_bas, train_adv])   #concatenate training and validation sets across Advanced and Basic
    valid = pd.concat([valid_bas, valid_adv])      

    return train, valid, testset