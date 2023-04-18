import pickle
import glob
import numpy as np
import pandas as pd
import sys
sys.path.append("../../")

from DanceProj1.DanceObj import Dance

# def get_data(path):
#     #allfilenames = glob.glob(f'{path}/*')         
#     #get lists of filenames per genre, basic and advanced

#     genrefilesBM = [glob.glob(f'{path}/g{i}_sBM*') for i in ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']]
#     genrefilesFM = [glob.glob(f'{path}/g{i}_sFM*') for i in ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']]

#     #initialize dictionaries for position arrays, per basdic/advanced per genre
#     genredata = {'Basic': {'Break':[], 'Pop':[], 'Lock':[], 'Midhop':[], 'LAhop':[], 'House':[], 'Waack':[],
#                     'Krump':[], 'Street Jazz':[], 'Ballet Jazz':[]}, 
#                 'Advanced': {'Break':[], 'Pop':[], 'Lock':[], 'Midhop':[], 'LAhop':[], 'House':[], 'Waack':[],}}
    

#     genres = ['Break', 'Pop', 'Lock', 'Midhop', 'LAhop', 'House', 'Waack', 'Krump', 'Street Jazz', 'Ballet Jazz']

#     for level in ['Basic', 'Advanced']:
#         for genre in genres:
#             for file in genrefilesBM[
        

            
#incomplete, working on it. Not sure the above lines are correct.