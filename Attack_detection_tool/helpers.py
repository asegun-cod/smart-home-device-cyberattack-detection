
import pandas as pd
import numpy as np
import pickle
# pathlib option
from pathlib import Path

def read_multifiles(basepath='/content/', datalist=[], pickle_file=True):
    '''
    read all CSV files in a folder, save them in a list and pickle them (if pickle_file is true)
    '''
    
    if pickle_file:  # create the pickled folder in case it doesn't exist
        new_path = 'picked/'
        Path(new_path).mkdir(exist_ok=True)
    for item in Path(basepath).iterdir(): 
        if item.is_file():  # filter out items that are directories
            print('processing:', item.name)
            try:
                dt = pd.read_csv(item,  parse_dates = ['Time']) # read the data for later pickle
            except:
                dt = pd.read_csv(item, encoding='latin1', parse_dates = ['Time'])
            
            dt['name'] = item.name[:-4]       # to identify the data later
            dt['Sequence Number'].replace([np.nan], '', inplace=True) 
            
            datalist.append(dt)
            if pickle_file:
                with open(basepath+new_path+item.name[:-4], 'wb') as f:    # save each read data into its file name
                  pickle.dump(dt, f)     # data to dump 
    # [print(n.name[0]) for n in datalist]   # view the arrangement of the files
    print('All files have now been loaded')
    return datalist
            
