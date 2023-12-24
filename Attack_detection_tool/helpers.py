## Helper Functions
# To extract flow metrics from traffic flow data and for the modeling

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
            print('Loading:', item.name)
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


    # PREPROCESSING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dat
import pickle
import re
from statistics import mode
# from pathlib import Path


# preprocessing tool
import sklearn
from sklearn.preprocessing import StandardScaler

# modeling tool
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score

def extract_tcp_flag(n):
    '''
    to extract TCP flag from the info column
    '''
    pattern = r"\d\[\w*\W*\w*]|\W\[\w*\W*\w*]"
    pattern2 = r"[A-Z]{3}"
    a = ''.join(re.split("\s", n))
    b = re.findall(pattern, a)
    p = re.findall(pattern2, b[0]) if b else ''
    p = ''.join([f[0] for f in p])
    return p

def get_tcp_flag(n):
    '''
    used to extract TCP flag from the "TCP Flags" column
    '''
    return ''.join(re.findall("[a-zA-Z]", n)) 

def scale_data (data, scaler):       
    scaled_X = scaler.transform(data) 
    return pd.DataFrame(scaled_X, columns=data.columns)

# def difference_fist_last(x):
#     return x.iloc[-1] - x.iloc[0]

# def strip_seconds (time):
#     'return the seconds and microseconds part of a timedelta object as a float (sec.microseconds)'
#     return round(float(str(time).split()[2].split(':')[2]),5)

# def  calc_diff (item):
#   "return as list the differences between two consecutive numbers in a list"
#   diff = item.rolling(window=2).apply(difference_fist_last)
#   return diff
  
def microsec_between_fist_last(x):
    return (x.iloc[-1] - x.iloc[0]).total_seconds() * 1000

def calc_time_diff (timer):
  "return as list the milliseconds difference between two consecutive timedelta object"
  t_diff = [0.00]
  _ = [t_diff.append((b-a).total_seconds() * 1000) for a, b in zip(timer, timer[1:])]
  return t_diff
    
# MODEL EVALUATION
def get_model_eval (y_true, y_pred, print_eval = False):  

    bacc = balanced_accuracy_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    if not print_eval:
        return {
        'Balance Accuracy': bacc, 
        # 'auc' : auc,
        'f1_score': f1}

    print('='*50)
    print('Balance Accuracy: ', bacc),
    # print('auc: ', auc)  
    print('f1_score: %.2f' %(f1*100))
    print('='*50)

# VISUALIZATION
def get_pca (df, scale=True): 
    pca = sklearn.decomposition.PCA(n_components=2)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)]) if scale else Pipeline([('pca', pca)])
    pca_df = pipe.fit_transform(df)
    print('PCA explained variance ratio:', pca.explained_variance_ratio_)
    return pca_df
                     
# Predicted label visualisation
def plot_pred_label (pca_data, y_pred, title = 'using label encoding'):
    plot = sns.scatterplot(x = pca_data[:,0], y = pca_data[:,1], 
                    hue=y_pred, alpha=None, 
                    palette= "tab10"
                   )
    plot.set(title = f"Predicted label visualisation ({title})")
    plt.show() 

# True label visualisation
def plot_true_label (pca_data, y_true, title = 'one-hot'):
    plot = sns.scatterplot(x = pca_data[:,0], y = pca_data[:,1], 
                    hue=y_true, alpha=None, 
                    palette= "tab10"
                   )
    plot.set(title = f"True label visualisation ({title})")
    plt.show()

# MAKE FLOW DATA
def extract_flow_metrics(rolling_df, flow_dt): 
    '''
        Extracting flow metrics from raw traffic data 
        
        return a dictionary of the metrics
    ''' 

    # transform the data
    rolling_df = rolling_df.replace(np.nan, '')   # to avoid the error presented by 'NaN'
    
    # TCP Flag variation
    try: 
        TCP_flags = list(map(get_tcp_flag, rolling_df['TCP Flags'])) 
    except:
        TCP_flags = list(map(extract_tcp_flag, rolling_df.Info)) 

    flow_dt['count_tcp_flags'].append(len(set(TCP_flags)))
    flow_dt['count_syn_flag'].append(''.join(TCP_flags).count('S'))
    flow_dt['count_ack_flag'].append(''.join(TCP_flags).count('A'))
    flow_dt['count_fin_flag'].append(''.join(TCP_flags).count('F'))
    flow_dt['count_rst_flag'].append(''.join(TCP_flags).count('R'))
    flow_dt['count_psh_flag'].append(''.join(TCP_flags).count('P'))

    # protocol variation
    flow_dt['count_icmp'].append(rolling_df.Protocol.tolist().count('ICMP'))
    flow_dt['count_udp'].append(rolling_df.Protocol.tolist().count('UDP'))
    flow_dt['count_tcp'].append(rolling_df.Protocol.tolist().count('TCP'))  
    flow_dt['count_tls'].append(rolling_df.Protocol.tolist().count('TLSv1.2'))   # track presense of encrypted packets
    flow_dt['count_ntp'].append(rolling_df.Protocol.tolist().count('NTP'))  
    flow_dt['count_dns'].append(rolling_df.Protocol.tolist().count('DNS'))  
    flow_dt['no_unique_prot'].append(len(rolling_df.Protocol.unique()))
    
    # flow_dt['ave_pack_IAT'].append(calc_diff(rolling_df.time_ff).mean())
    # flow_dt['flow_dur'].append(difference_fist_last(rolling_df.time_ff)  
    
    flow_dt['ave_pack_IAT'].append(np.mean(calc_time_diff(rolling_df.Time)))
    flow_dt['flow_dur'].append(microsec_between_fist_last(rolling_df.Time))

    flow_dt['no_unique_pl'].append(len(rolling_df.Length.unique()))
    if mode(rolling_df["Sequence Number"]) == '':
        flow_dt['sn_type'].append(-1) 
    elif mode(rolling_df["Sequence Number"]) > 0:
        flow_dt['sn_type'].append(1) 
    else:
        flow_dt['sn_type'].append(-1)
        
    return flow_dt 

def make_training_data(df, device_ipadd, attack_ipadd= None, roller=47, step=2):
    '''
    Group specified numbers of packets together, extract flow metric and 
    make the flow data for training the model
    '''  
    # Verify that the required columns are present and correctly named
    required_col = ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'Sequence Number', 'Info']
    assert all([i in df.columns.tolist() for i in required_col]), f"The following columns are required: {[i for i in required_col if i not in list(df.columns)]}"
    
    flow_dt = {'pkt_start':[], 'pkt_end':[], 'flow_dur':[], 'ave_pack_IAT':[],'count_tcp_flags':[], 
                'count_syn_flag':[], 'count_ack_flag':[], 'count_fin_flag':[], 'count_rst_flag':[], 'count_psh_flag':[], 
                'count_tcp':[], 'count_tls':[],'count_icmp':[], 'count_udp':[],  'count_ntp':[], 'count_dns':[], 
                'no_unique_prot':[], 'no_unique_pl':[], 'sn_type':[]
            }
    
    df.Protocol.replace({'SSH':'TLSv1.2'}, inplace=True) # make all encryption packets have consistent names

    # df.rename(columns = {'Time since reference or first frame':'time_ff'}, inplace=True) #to conveniently use the column name

    # if device_ipadd: data = df.query(f'Source == "{device_ipadd}" | Destination == "{device_ipadd}"')  # to filter out background noise
    if attack_ipadd == None:
        df = df.query(f'Destination == "{device_ipadd}"').reset_index(drop=True)  # to filter out background noise
    else:
        df = df.query(f'Source == "{attack_ipadd}" & Destination == "{device_ipadd}"').reset_index(drop=True)  # to filter out background noise
    assert  len(df) > 0, 'Confirm that the correct device IP address is provided. All the traffic count been filtered out as background noise.'
    # print(df) # for debugging

    flow_id = 1
    start = 0
    roller = roller
    step = step
    print('Sliding Window set at', 1+roller+step)

    for r in range(0, len(df), step):                               
        if (r+roller > len(df)): 
    #         print (start, ':', len(df))
            flow_dt['pkt_start'].append(start)
            flow_dt['pkt_end'].append(len(df))
            rolling_df = df[start:end]
        else :
            if r == 0:
                continue 
            else:
                end = r+roller
    #             print(start, ':', r+roller)
                rolling_df = df[start:end]
                flow_dt['pkt_start'].append(start)
                flow_dt['pkt_end'].append(end)
                
        flow_dict = extract_flow_metrics(rolling_df, flow_dt)
        start = r
        flow_id+= 1
    print('Done extracting flow data')
    return pd.DataFrame.from_dict(flow_dict).iloc[:,0:]

def make_modeling_data(datalist, device_ipadd, attack_ipadd, df_name='', roller=7, step=2, save_to_csv=True):
    '''
    Extract flow data from series of raw traffic data and combine all 
    the data together to make a single data for trainig or testing our model
    '''
    all_flow_dt = pd.DataFrame()
    for d in datalist:
        if d.name.unique()[0] == 'Normal01': 
            flow_data = make_training_data(d, device_ipadd="192.168.0.101", roller=roller, step=step)
        elif d.name.unique()[0] == 'Normal02':
            flow_data = make_training_data(d, device_ipadd="rcr-663.local", roller=roller, step=step)
        else:
            flow_data = make_training_data(d, device_ipadd, attack_ipadd, roller=roller, step=step)
        flow_data['label'] = d.name.unique()[0]
    #     flow_data.to_csv(f'{d.name.values[0]}_flow.csv')
        all_flow_dt=pd.concat([all_flow_dt, flow_data], ignore_index=True)
        print(f'Done with {d.name.unique()[0]}')    
    print('Done with all the data')
    if save_to_csv: all_flow_dt.to_csv(f'../data/Training_data/{df_name}_v1.0.csv')
    return all_flow_dt

def make_attack_model (flow_data, clf, scale=False, plot_eval=False):
    '''
        Combine series of steps to make a  model for the attack detection. 
        It print the model's confusion matrix plot and some other metrics. 
        
        return a list of the trained modeln and the scaler model 
        used for scaling the traaining features
    '''
    #2 Preprocess data
    y = flow_data['label']
    X = flow_data.drop(columns = ['pkt_start','pkt_end', 'label'])  # drop dummy column and the actual label 
    # X['av_sn'].replace([np.nan], -1, inplace=True)   # encode flows with no average sequece number (nan) with -1

    #3 create scaler and scale the data
    scaler = StandardScaler().fit(X)  
    if scale: 
        X = scale_data(X, scaler)

    #4 modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True, stratify=y)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #5 evaluate model

    # Visualise confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

    met = pd.DataFrame(np.round(precision_recall_fscore_support(y_test, y_pred), 3))
    met.columns =model.classes_
    met.index = ['precision', 'recall', 'fscore', 'support']
    print(met)
    get_model_eval(y_test, y_pred, print_eval=True)
    score = get_model_eval(y_test, y_pred)

    if plot_eval:
        # visualization
        pca_le = get_pca(X_test, scale=False) if scale else get_pca(X_test) # PCA
        plot_pred_label (pca_le, y_pred, title = 'Label-encoding') # predicted label
        # plot_true_label(pca_le, y_test, 'Label-encoding')  # true label
   
    return model, scaler, score

def attack_detector (df, trained_model, scaler, device_ipadd=None, roller=7, step=2):
    '''
        Predict if a tuple of `roller + step + 1` packets is an attack or normal flow 
        by extracting flow metric from the `roller + step + 1` packet tuple and 
        using the loaded trained model (created using `make_attack_model`) for the prediction.
    '''

    if device_ipadd:
        data = df.query(f'Destination == "{device_ipadd}"') # to filter out background noise 
    else: 
        data = df 

    assert  len(data) > 0, 'Confirm that the correct device IP address is provided. All the traffic count been filtered out as background noise.'
    # print(data) # for debugging

    data.Protocol.replace({'SSH':'TLSv1.2'}, inplace=True) # make all encryption packets have consistent names

    # convert Time column in strings to datatime 
    if data.Time.dtype == 'O':
        data.Time = pd.to_numeric(data.Time)
        data.Time = data.Time.apply(dat.fromtimestamp)
        
    
#1  Extract flow data ------->flow_df = make_flow_data(traffic_df)
    flow_id = 1
    start = 0
    atk_name = []
    atk_str  = []
    atk_end = []
    atk_mode = []
    for r in range(0, len(data), step):        
        flow_dt = {'pkt_start':[], 'pkt_end':[], 'flow_dur':[], 'ave_pack_IAT':[],'count_tcp_flags':[], 
                'count_syn_flag':[], 'count_ack_flag':[], 'count_fin_flag':[], 'count_rst_flag':[], 'count_psh_flag':[], 
                'count_tcp':[], 'count_tls':[],'count_icmp':[], 'count_udp':[],  'count_ntp':[], 'count_dns':[], 
                'no_unique_prot':[], 'no_unique_pl':[], 'sn_type':[]
            }
        
        # get chunk of traffic packets (based on the defined window (roller+step+1))
        if (r+roller > len(data)): 
            flow_dt['pkt_start'].append(start)
            flow_dt['pkt_end'].append(len(data))
            rolling_df = data[start:end]
        else :
            if r == 0:
                continue 
            else:
                end = r+roller
                # print(start, ':', r+roller)
                rolling_df = data[start:end]
                flow_dt['pkt_start'].append(start)
                flow_dt['pkt_end'].append(end)
        # print(rolling_df) # for debugging

        # extract flow metrics from the chunk of traffic packets  
        flow_dict = extract_flow_metrics(rolling_df, flow_dt)
        
        # convert the flow metrics data into a dataframe
        flow_dt = pd.DataFrame.from_dict(flow_dict) 
        # print('='*10, f'flow {flow_id} : packet {start} -- packet {end}', '='*10)  # for debugging purpose
        # print(flow_dt) # for debugging purpose

#2      Preprocess the flow data
        feature = flow_dt.drop(columns = ['pkt_start','pkt_end'])  # drop dummy column and the actual label 
        # feature['av_sn'].replace([np.nan], -1, inplace=True)   # encode flows with no average sequece number (nan) with -1
        
#3      scale the flow data
        feature = scale_data(feature, scaler)
            
        packet_info = flow_dt.loc[:,['pkt_start','pkt_end']]
#         print('='*50, )    ##for debugging purpose
#         print(feature)     ##for debugging purpose

#5      make prediction
        pred = trained_model.predict(feature)[0]
        # print (pred)   ##for debugging purpose

#6      Take action based on the prediction (traffic flow type)
        if pred != 'Normal':
            atk_name.append(pred)                                                         # For monitoring or later analysis
            atk_str.append(packet_info.pkt_start[0])                                      # For monitoring or later analysis
            atk_end.append(packet_info.pkt_end[0])                                        # For monitoring or later analysis
                                                                                    
            try:                                                                      
                atk_mode.append(mode([x for x in rolling_df.Protocol if x != 'TLSv1.2']))   # to guess the attack type using the most common protocol type after eliminating TLS protocols
            except:                                                                     
                atk_mode.append('unknown')                   # For monitoring or later analysis
            print( f'"{pred}" Attack(attack mode - {atk_mode}) detected between packet ==> {packet_info.pkt_start[0]} and {packet_info.pkt_end[0]} (original index {rolling_df.index[0]+1} : {rolling_df.index[-1]+2})\n\
                 stop server NOW')
            break    # un/comment if you need to stop detection after first attack has been detected
        # else:                                                                        # For monitoring 
        #     print('"Normal flow"')                                                     
        # print('='*50, f'flow {flow_id} : start {end}', '='*50)  
        start = r
        flow_id+= 1
    return atk_name, atk_str, atk_end, atk_mode