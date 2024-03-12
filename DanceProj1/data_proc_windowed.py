
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

    #note we deleted the following IDs from the keypoints folder, because of HTTP errors:
    #'gJB_sFM_cAll_d08_mJB3_ch11', 'gBR_sFM_cAll_d05_mBR5_ch14', 'gJS_sFM_cAll_d02_mJS0_ch08', 'gWA_sFM_cAll_d26_mWA2_ch10']
    #gBR_sFM_cAll_d06_mBR5_ch19, 'gBR_sFM_cAll_d04_mBR4_ch07', gJS_sFM_cAll_d01_mJS1_ch02, gJS_sFM_cAll_d03_mJS0_ch01,
    #gBR_sFM_cAll_d06_mBR4_ch20

    #deleted the following IDs from the keypoints folder, because of pose tracking errors
    #gJS_sFM_cAll_d01_mJS0_ch01.mp4 

    #deleted the following (basic) pieces from the keypoints folder, because of index errors

    #manually add gBR_sBM_cAll_d04_mBR0_ch02 to the csv, for the example clip 

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

def create_windows(sequence, window_sizes=[120], stepsize=30):
    """
    Create windows, i.e. sub-sequences from the given 3D pose sequence.
    
    Parameters:
    - sequence (numpy array): The input 3D pose sequence with shape (joints, frames, dimensions).
    - window_sizes (list): List of window sizes to use for creating sub-sequences.
    
    Returns:
    - windows_dict (dict): Dictionary where keys are window sizes and values are lists of sub-sequences.
    """
    # Initialize dictionary to store sub-sequences for each window size
    windows_dict = {}
    
    # Total number of frames in the sequence
    total_frames = sequence.shape[1]
    
    # Loop through each specified window size
    for win_size in window_sizes:
        
        # Initialize list to store sub-sequences for this window size
        sub_sequences = []
        
        # Skip this window size if the sequence is shorter than the window
        if total_frames < win_size:
            continue
        
        # Create sub-sequences by sliding the window
        for start in range(0, total_frames - win_size + 1, stepsize):  # Sliding step of 30 frames
            end = start + win_size
            sub_seq = sequence[:, start:end, :]
            sub_sequences.append(sub_seq)
        
        # Add to the dictionary
        windows_dict[win_size] = sub_sequences
    
    return windows_dict

# function for deriving the features from pose sequence data. No windows here.     
# def data_to_features(dataBM, dataFM, sparse=False):

#     featuresBM = []                     # list for feature-dictionaries for all basic dances    
#     errors = []

#     for genre in dataBM:
#         for i in range(len(dataBM[genre])):
#             dance = Dance(dataBM[genre][i][0], 1/60)   #create Dance object for each basic dance
#             dance.genre = genre                             #add genre
#             dance.id = dataBM[genre][i][1]             #add id
#             try:
#                 dance.get_features(sparse=sparse)    #do a try / except here, print the ID where the error occured and then continue    
#             except IndexError:
#                 print(dance.id)
#                 errors.append(dance.id)
#                 continue
#             featuresBM.append(dance.features)                   #add feature-dict to list
    
#     featuresFM = []                     # repeat for all advanced dances
#     for genre in dataFM:
#         for i in range(len(dataFM[genre])):
#             dance = Dance(dataFM[genre][i][0], 1/60)
#             dance.genre = genre
#             dance.id = dataFM[genre][i][1]
#             try:
#                 dance.get_features(sparse=sparse)
#             except IndexError:
#                 print(dance.id)
#                 errors.append(dance.id)
#                 continue
#             featuresFM.append(dance.features)
    
#     print('there were index errors on', len(errors), 'dances')

            
#     #turn these into dataframes
#     dfBM = pd.DataFrame(featuresBM)
#     dfFM = pd.DataFrame(featuresFM)            
            
#     return dfBM, dfFM

# function for deriving the features from pose sequence data on all the windows for each sequence
def data_to_windowed_features(dataBM, dataFM, sparse=False, window_sizes=[120]):

    featuresBM = []  # list for feature-dictionaries for all basic dances    
    featuresFM = []  # list for feature-dictionaries for all advanced dances
    errors = []

    for dataset, features in [(dataBM, featuresBM), (dataFM, featuresFM)]:
        for genre in dataset:
            for i in range(len(dataset[genre])):
                # Create windows using the create_sliding_windows function
                windows_dict = create_windows(dataset[genre][i][0], window_sizes)

                for win_size, windows in windows_dict.items():
                    for window_index, window in enumerate(windows):
                        dance = Dance(window, 1/60)  # Create Dance object for each window
                        dance.genre = genre  # Add genre
                        dance.id = dataset[genre][i][1]  # Add id
                        dance.window_size = win_size  # Add window size
                        dance.features["window size"] = win_size
                        dance.features["window number"] = window_index
                        try:
                            dance.get_features(sparse=sparse)
                        except IndexError:
                            print(f"Error on ID {dance.id}, Window Size {win_size}")
                            errors.append((dance.id, win_size))
                            continue
                        features.append(dance.features)

    print(f'There were index errors on {len(errors)} windows')

    # Turn these into dataframes
    dfBM = pd.DataFrame(featuresBM)
    dfFM = pd.DataFrame(featuresFM)

    return dfBM, dfFM
        
     
def traintestval_split(dfBasic, dfAdvanced, testfrac_adv=.51, testfrac_bas=0, valfrac_adv_nonT=0, valfrac_bas=0):

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

def traintest_split_basic(dfBasic, dfAdvanced):
    testset = pd.DataFrame(columns=dfBasic.columns)  # Initialize testset
    train = pd.DataFrame(columns=dfBasic.columns)  # Initialize trainset

    # Calculate number of Basic pieces to sample for the test set
    num_basic_test = 103

    # Assuming dfBasic is already shuffled or you're okay with random sampling
    testset = dfBasic.sample(n=num_basic_test, random_state=1)  # Sample 103 Basic pieces for the test set
    
    # The rest of the Basic sequences are used for training
    train_bas = dfBasic.drop(testset.index)  # Exclude test set from training set

    # Include all Advanced sequences in the training set, assuming no need to differentiate here based on your description
    train_adv = dfAdvanced
    
    # Combine Basic and Advanced sequences for the final training set
    train = pd.concat([train_bas, train_adv])

    return train, testset


 