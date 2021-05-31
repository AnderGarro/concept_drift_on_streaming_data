from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy import interpolate
from datetime import date, timedelta
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering, DBSCAN
from sklearn import svm
import pandas as pd
import time
import statistics
import math



def adjust_time(df):
    time_list = df.time.values
    item_0 = time_list[0]
    time_delta = []
    for item in time_list:
        try:
            a = datetime.datetime.strptime(item[:-1], '%Y-%m-%dT%H:%M:%S.%f')
        except:
            a = datetime.datetime.strptime(item[:-1], '%Y-%m-%dT%H:%M:%S')
        try:
            b = datetime.datetime.strptime(item_0[:-1], '%Y-%m-%dT%H:%M:%S.%f')
        except:
            b = datetime.datetime.strptime(item_0[:-1], '%Y-%m-%dT%H:%M:%S')
        time_delta.append(a-b)
    df["time_delta"] = time_delta
    return df

def filter_data(df):
    tag_list = df["tag_id"].unique()
    n = 0
    m = 0
    for tag_id in tag_list:
        # Generate time_delta
        df_item = df[(df["tag_id"] == tag_id) & (df["tag_var"] == "TP")]
        df_item = adjust_time(df_item)

        # Filter data elapsed more than 2.5s
        sec = df_item["time_delta"].values[-1] / np.timedelta64(1000000000, 'ns')

        # Filter superposed data
        tag_pos_bool = df_item["tag_pos"].values

        # Apply filters and store filtered data in df_filtered
        if sec < 2.5 and (False not in (sorted(tag_pos_bool) == tag_pos_bool)):
            if m == 1:
                df_filtered = pd.concat([df_filtered,df_item], axis=0) 
            else:
                df_filtered = pd.concat([df_item], axis=0) 
            m=1

        # Store whole dataset in df_all
        if n == 1:
            df_all = pd.concat([df_all,df_item], axis=0) 
        else:
            df_all = pd.concat([df_item], axis=0) 
        n=1
    return df_all,df_filtered

def get_tag_id_list(df):
    tag_id_list   = df[df["tag_var"] == "TP"]["tag_id"].unique()
    tag_var_list  = df["tag_var"].unique()
    tag_zone_list = df["tag_zone"].unique() 
    return tag_id_list,tag_var_list,tag_zone_list
def extract_metrics(df_item,value):
    values   = df_item[(df_item["tag_var"] == value)]["value"].values
    mean_f   = mean(values)
    std_f    = std(values)
    features = [mean_f,std_f]
    return features

def extract_ts(df_item,tag_id,tag_var):
    df   = df_item[(df_item["tag_id"] == tag_id)]
    df   = df[(df["tag_var"] == tag_var)]
    return df

def extract_features(timeserie, fs, dur, item):
    # Características temporales
    maximum_value=maximum_ts(timeserie)
    minimum_value=minimum_ts(timeserie)
    mean_value=mean_ts(timeserie)
    peak_to_peak_value=peak_to_peak(timeserie)
    trimmed_mean_value=trimmed_mean(timeserie)
    variance_value=variance_ts(timeserie)
    standard_deviation_value=standard_deviation(timeserie)
    mean_abs_deviation_value=mean_abs_deviation(timeserie)
    median_abs_deviation_value=median_abs_deviation(timeserie)
    crest_factor_value=crest_factor(timeserie)
    Peak2RMS_value=Peak2RMS(timeserie)
    skewness_value=skewness(timeserie)
    kurtosis_value=kurtosis(timeserie)
    shape_factor_value=shape_factor(timeserie)
    rms_value=rms(timeserie)

    # Características ámbito frecuencial. Para extraerlas hacemos la fft de la timeserie
    X, freq = frequency_conversion(timeserie, fs, dur)

    mean_frequency_value=mean_frequency(X, freq)
    frequency_center_value=frequency_center(X, freq)
    rms_frequency_value=rms_frequency(X, freq)
    sd_frequency_value=sd_frequency(X, freq)
    largest_freq_amp_value=largest_freq_amp(X, freq)
    largest_freq_idx_value=largest_freq_idx(X, freq)
    largest_sideband_amp_value=largest_sideband_amp(X, freq)
    sideband_index_value=sideband_index(X, freq)
    sideband_level_factor_value=sideband_level_factor(X, freq, timeserie)
    figure_of_merit_value=figure_of_merit(timeserie)
    features = {
                    "id": item,
                    "maximum": maximum_value,
                    "minimum": minimum_value,
                    "mean": mean_value,
                    "peak_to_peak": peak_to_peak_value,
                    "trimmed_mean": trimmed_mean_value,
                    "variance": variance_value,
                    "standard_deviation": standard_deviation_value,
                    "mean_abs_deviation": mean_abs_deviation_value,
                    "median_abs_deviation": median_abs_deviation_value,
                    "crest_factor": crest_factor_value,
                    "Peak2RMS": Peak2RMS_value,
                    "skewness": skewness_value,
                    "kurtosis": kurtosis_value,
                    "shape_factor": shape_factor_value,
                    "rms": rms_value,
                    "mean_frequency": mean_frequency_value,
                    "frequency_center": frequency_center_value,
                    "rms_frequency": rms_frequency_value,
                    "sd_frequency": sd_frequency_value,
                    "largest_freq_amp": largest_freq_amp_value,
                    "largest_freq_idx": largest_freq_idx_value,
                    "largest_sideband_amp": largest_sideband_amp_value,
                    "sideband_index": sideband_index_value,
                    "sideband_level_factor": sideband_level_factor_value,
                    "figure_of_merit": figure_of_merit_value
    }
    return features

def mean(values):
    return values.mean()

def std(values):
    return values.std()

# Función para convertir la señal al dominio frecuencial
def frequency_conversion(timeserie, fs, dur):
    N = dur * fs
    X = np.fft.fft(timeserie)
    freq = np.linspace (0.0, fs/2, int (N/2))
    X = 1/N * np.abs (X[0:np.int (N/2)])
    return X, freq

# Funciones para cada una de las características
def maximum_ts(timeserie):
    return np.max(timeserie)

def minimum_ts(timeserie):
    return np.min(timeserie)

def mean_ts(timeserie):
    return np.mean(timeserie)

def peak_to_peak(timeserie):
    return abs(np.max(timeserie))+abs(np.min(timeserie))

def trimmed_mean(timeserie):
    return stats.trim_mean(timeserie, 0.25)

def variance_ts(timeserie):
    return np.var(timeserie)

def standard_deviation(timeserie):
    return np.std(timeserie)

def mean_abs_deviation(timeserie):
    return sum(np.abs(timeserie-np.mean(timeserie)))/len(timeserie)

def median_abs_deviation(timeserie):
    return sum(np.abs(timeserie-np.median(timeserie)))/len(timeserie)

def crest_factor(timeserie):
    return np.max(timeserie)/rms(timeserie)

def Peak2RMS(timeserie):
    return np.max(np.abs(timeserie))/rms(timeserie)

def skewness(timeserie):
    return stats.skew(timeserie)

def kurtosis(timeserie):
    return stats.kurtosis(timeserie, fisher=False)

def shape_factor(timeserie):
    return rms(timeserie)/(np.sum(np.abs(timeserie))*1/len(timeserie))

def rms(timeserie):
    return np.sqrt(np.mean(timeserie**2))

def mean_frequency(X, freq):
    return np.mean(X)

def frequency_center(X, freq):
    return sum(freq*X)/sum(X)

def rms_frequency(X, freq):
    return np.sqrt(sum((freq**2)*X)/sum(X))

def sd_frequency(X, freq):
    return np.std(X)

def largest_freq_amp(X, freq):
    peaks, amp = signal.find_peaks(X, height=0)
    amp = amp["peak_heights"]
    return np.max(amp)

def largest_freq_idx(X, freq):
    peaks, amp = signal.find_peaks(X, height=0)
    amp = amp["peak_heights"]
    return freq[peaks[np.argmax(amp)]]

def largest_sideband_amp(X, freq):
    peaks, amp = signal.find_peaks(X, height=0)
    amp = amp["peak_heights"]
    max_values = amp.argsort()[-2:][::-1]
    second_max_value_idx = max_values[1]
    return amp[second_max_value_idx]

def sideband_index(X, freq):
    peaks, amp = signal.find_peaks(X, height=0)
    amp = amp["peak_heights"]
    max_values = amp.argsort()[-2:][::-1]
    second_max_value_idx = max_values[1]
    return freq[peaks[second_max_value_idx]]

def sideband_level_factor(X, freq, timeserie):
    return largest_sideband_amp(X, freq)/rms(timeserie)

def figure_of_merit(timeserie):
    return peak_to_peak(timeserie)/sum(timeserie)

def get_complex_features(df,tag_id_list):# Frecuencia muestreo (Hz)
    fs = 200
    # Duración de la señal
    dur = 1
    data_features = []
    n = 0
    for item in tag_id_list:
        n = n +1
        try:
            #print(item,n,len(tag_id_list),"OK")
            serie = extract_ts(df,item,"TP")
            ts = serie.value
            serie_features = extract_features(ts, fs, dur, item)
            data_features.append(serie_features)
            del serie_features
        except:
            #print(item,n,len(tag_id_list),"Too Short or other error")
            continue
    df_features = pd.DataFrame(data_features)
    return df_features

def read_original():
    df_draw     = pd.read_csv("/data/df_draw.csv")
    df_filtered = pd.read_csv("/data/df_filtered.csv")
    return df_draw,df_filtered
    
def get_slope(y_crop,variable):
    X = np.array(y_crop["id"].values).reshape(-1, 1)  # values converts it into a numpy array
    Y = np.array(y_crop[variable].values).reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    return linear_regressor

def insert_noise(df,variable):
    std  = np.std(df[variable])
    noise = []
    for i in list(df[variable].values):
        noise.append(np.random.normal(i, std, 1)[0])
    return noise

def make_predictions(df, variable, num):
    # Stick to variable
    y_crop = df[["id",variable]]
    
    # Get slope
    linear_regressor = get_slope(y_crop,variable)
    
    # Get predictions from slope
    X = np.linspace(y_crop["id"].values[0], y_crop["id"].values[-1], num=num, endpoint=True).reshape(-1,1)
    Y_fit  = linear_regressor.predict(X)  # make predictions
    y_pred = pd.DataFrame(data=np.array([X.reshape(num),Y_fit.reshape(num)]).T,columns=["id",variable])
    
    # Insert noise on pred
    noise = insert_noise(y_pred,variable)
    y_pred[variable+"-noise"] = noise
    return y_crop,y_pred

def make_predictions_on_time(df, variable, time):
    # Stick to variable
    y_crop = df[["id",variable]]
    
    # Get slope
    linear_regressor = get_slope(y_crop,variable)
    
    id_point = y_crop["id"].values[0] + time*1000

    # Get predictions from slope
    X = np.linspace(id_point, id_point, num=1, endpoint=True).reshape(-1,1)
    #X = (id_point).reshape(-1,1)
    Y_fit  = linear_regressor.predict(X)  # make predictions
    y_pred = pd.DataFrame(data=np.array([X.reshape(1),Y_fit.reshape(1)]).T,columns=["id",variable])
    
    # Insert noise on pred
    noise = insert_noise(y_pred,variable)
    y_pred[variable+"-noise"] = noise
    return y_crop,y_pred

def get_mean(df_filtered):
    id_list = df_filtered["tag_id"].unique()
    values_list = []
    for item in id_list:
        df_item = df_filtered[df_filtered["tag_id"] == item ]
        values  = list(df_item["value"].values)
        values_list.append(values)
    df_data = pd.DataFrame(values_list, index=id_list).fillna(0)
    df_data.loc['mean'] = df_data.mean()
    mean_values = df_data.iloc[0:20].mean()[0:189]
    max_mean_values,min_mean_values,mean_mean_values = max(mean_values),min(mean_values),sum(mean_values)/len(mean_values)
    return max_mean_values,min_mean_values,mean_mean_values,mean_values

def get_synthetic(df_orginal,df_filtered,num = 1000, noise_amp=40):

    columns = list(df_orginal.columns)
    columns.remove("id")
    columns.remove("state")
    columns.remove("mean")
    
    y_crop, y_pred = make_predictions(df_orginal, "mean", num)
    for column in columns:
        y_crop_2, y_pred_2 = make_predictions(df_orginal, column, num)
        y_pred = y_pred.join(y_pred_2.set_index("id"),on="id")
        y_crop = y_crop.join(y_crop_2.set_index("id"),on="id")
        
    max_mean_values,min_mean_values,mean_mean_values,mean_values = get_mean(df_filtered)
    
    y_pred = y_pred[["id","maximum","minimum","mean"]].set_index("id")
    for i in range(189):
        a = (np.random.random(1000)-0.5)*2
        y_pred[str(i)] = mean_values[i]-mean_mean_values+y_pred["mean"]*(1+a/noise_amp)
    return y_pred

def make_synthetic(create=True):
    # Generate Original Base to replicate
    print("Generating Original data")
    if create:
        df_original,df_filtered = get_original()
    else:
        df_original,df_filtered = read_original()
    print("Generating Original data: DONE")
    
    # Generate synthetic data out of the original base
    print("Generating Synthetic data")
    y_pred                  = get_synthetic(df_original,df_filtered)
    print("Generating Synthetic data: DONE")
    return y_pred

def get_curve(time, noise_amp=40, create=False):
    # Generate Original Base to replicate
    print("Generating Original data")
    if create:
        df_original,df_filtered = get_original()
    else:
        df_original,df_filtered = read_original()
    print("Generating Original data: DONE")
    
    # Generate synthetic data out of the original base
    print("Generating Synthetic data")
    y_pred                  = get_synthetic_curve(df_original,df_filtered, time, noise_amp)
    print("Generating Synthetic data: DONE")
    return y_pred

def get_synthetic_curve(df_original,df_filtered,time, noise_amp):
    columns = list(df_original.columns)
    columns.remove("id")
    columns.remove("state")
    columns.remove("mean")

    #time = 1.65*60*60*24*365.25

    y_crop, y_pred = make_predictions_on_time(df_original, "mean", time)
    for column in columns:
        y_crop_2, y_pred_2 = make_predictions_on_time(df_original, column, time)
        y_pred = y_pred.join(y_pred_2.set_index("id"),on="id")
        y_crop = y_crop.join(y_crop_2.set_index("id"),on="id")
        
    max_mean_values,min_mean_values,mean_mean_values,mean_values = get_mean(df_filtered)
    
    y_pred = y_pred[["id","maximum","minimum","mean"]].set_index("id")
    for i in range(189):
        a = (np.random.random(1)-0.5)*2
        y_pred[str(i)] = mean_values[i]-mean_mean_values+y_pred["mean"]*(1+a/noise_amp)
    return y_pred

