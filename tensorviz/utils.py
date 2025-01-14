import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# csvファイルのパスを受け取り、データフレームの上位5行を表示する関数
def show_head(csv_path):
    df = pd.read_csv(csv_path)
    return df.head()

# csvファイルのパスを受け取り、データフレームの情報を表示する関数
def show_info(csv_path):
    df = pd.read_csv(csv_path)
    return df.info()

# Convert a DataFrame (list) to tensor time series
# (Created by Koki Kawabata)
#
# input: 
#   df (pandas.DataFrame):  A list of discrete events
#   time_key (str):         A column name of timestamps
#   facets (list):          A list of column names to make tensor timeseries
#   values (str):           A column name of target values (optional)
#   sampling_rate (str):    A frequancy for resampling, e.g., "7D", "12H", "H"
#   start_date (str):       A start date for slicing
#   end_date (str):         An end date for slicing
#   scaler (str):           A scaler for scaling the tensor [None, "full", "each"]
#
# output: 
#   tensor time series
#
def df2tts(df, time_key, facets, values=None, sampling_rate="D", start_date=None, end_date=None, scaler=None):
    df[time_key] = pd.to_datetime(df[time_key])
    if start_date is not None: df = df[lambda x: x[time_key] >= pd.to_datetime(start_date)]
    if end_date is not None: df = df[lambda x: x[time_key] <= pd.to_datetime(end_date)]

    # Print the information of the dataset
    print("[Start Date]:", df[time_key].min())
    print("[End Date]:", df[time_key].max())
    for mode in facets:
        print(mode, ":", sorted(list(set(df[mode]))))

    tmp = df.copy(deep=True)
    shape = tmp[facets].nunique().tolist()
    if values == None: values = 'count'; tmp[values] = 1
    tmp[time_key] = tmp[time_key].round(sampling_rate) # Round the timestamp

    print("Tensor:")
    print(tmp.nunique()[[time_key] + facets])

    grouped = tmp.groupby([time_key] + facets).sum()[[values]]
    grouped = grouped.unstack(fill_value=0).stack()
    grouped = grouped.pivot_table(index=time_key, columns=facets, values=values, fill_value=0)

    tts = grouped.values
    tts = np.reshape(tts, (-1, *shape))

    if scaler=="full":
        # Min-Max scaling
        tts =  minmax_scale(tts.reshape((-1, 1))).reshape(tts.shape)
    elif scaler=="each":
        # Min-Max scaling for each series
        tts = min_max_scale_tensor(tts)

    return tts

def min_max_scale_np(array):
    min_value = array.min()
    max_value = array.max()
    return (array - min_value) / (max_value - min_value)

def min_max_scale_tensor(data):
    query_size = data.shape[1]
    geo_size = data.shape[2]
    ret = np.zeros(shape=data.shape)
    for i in range(query_size):
        for j in range(geo_size):
            ret[:,i,j] = min_max_scale_np(data[:,i,j])
    return ret


def viz_tts_slice(tts, index=None, slice_mode=None):
    if slice_mode==1:
        entity_size = tts.shape[1]
        for entity_index in range(entity_size):
            plt.figure(figsize=(20, 5))
            if index:
                plt.plot(tts[:, entity_index, index])
            else:
                plt.plot(tts[:, entity_index, :])
            plt.show()
            plt.close()

    elif slice_mode==2:
        entity_size = tts.shape[2]
        for entity_index in range(entity_size):
            plt.figure(figsize=(20, 5))
            if index:
                plt.plot(tts[:, :, entity_index])
            else:
                plt.plot(tts[:, :, entity_index])
            plt.show()
            plt.close()


def save_tts_slice(tts, save_path, index=None, slice_mode=None):
    if slice_mode==1:
        entity_size = tts.shape[1]
        for entity_index in range(entity_size):
            plt.figure(figsize=(20, 5))
            if index:
                plt.plot(tts[:, entity_index, index])
            else:
                plt.plot(tts[:, entity_index, :])
            plt.savefig(save_path + f"/mode1_{entity_index}.png")
            plt.close()

    elif slice_mode==2:
        entity_size = tts.shape[2]
        for entity_index in range(entity_size):
            plt.figure(figsize=(20, 5))
            if index:
                plt.plot(tts[:, :, entity_index])
            else:
                plt.plot(tts[:, :, entity_index])
            plt.savefig(save_path + f"/mode2_{entity_index}.png")
            plt.close()
