import warnings
import datetime

import os
import sys

import pandas as pd
import numpy as np
import collections
import sys
import psycopg2
from psycopg2.extras import execute_values

from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score
from urllib.parse import urlparse

import time
import requests
import statistics
from scipy import stats, signal
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def connect_postgress():
    """ open conexion to the PostgreSQL database"""
    conn = psycopg2.connect(
        host="postgres-data",
        port=5432,
        database="sc",
        user="postgres",
        password="postgres")
    return conn

def insert_values(table_name,values,sql):
    """ insert valeus """
    conn = None
    vendor_id = None
    print(sql)
    try:
        # read database configuration
        # params = config()
        # connect to the PostgreSQL database
        conn = connect_postgress()
        #conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        # cur.execute(sql, (values,))
        # execute the SQL statement
        execute_values(cur, sql, values)

        # get the generated id back
        #cur.fetchone()
        # cur.fetchall()
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        id_values = None
    finally:
        if conn is not None:
            conn.close()

def get_data(table_name,sql):
    """ query data """
    conn = None
    try:
        conn = connect_postgress()
        cur = conn.cursor()
        cur.execute(sql)
        rowcount = cur.rowcount
        row = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return row,rowcount

def read_data(sql):
    rows, rowcount = get_data("ventas",sql)
    return rows

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

def get_features(df):# Frecuencia muestreo (Hz)
    fs = 50
    # Duración de la señal
    dur = 1
    data_features = []
    n = 0
    for item in range(len(df)):
        n = n +1
        ts = df.iloc[item].values
        serie_features = extract_features(ts, fs, dur, item)
        data_features.append(serie_features)
        del serie_features
    df_features = pd.DataFrame(data_features)
    return df_features

def get_max_p_value(machine):
    # last value on a machine
    sql = """
    SELECT a.TIME
    FROM p_value a
    INNER JOIN(
        SELECT max(TIME) as TIME
        FROM p_value
        WHERE MACHINE = """+str(machine)+"""
    ) b on a.TIME = b.TIME WHERE a.MACHINE = """ +str(machine)
    value = read_data(sql)
    return value[0][0]

def compute_p_value(machine):
    
    # READ PROFILES
    sql = "SELECT TIME,DATA FROM profiles where machine = "+str(machine)+" ORDER BY time"
    values = read_data(sql)

    # TRANFORM DATA TO LIST OF FLOATS
    values_pro = []
    times = []
    for item in values:
        times.append(item[0])
        values_pro.append([float(s) for s in item[1][2:-2].split(",")])
    df = pd.DataFrame(values_pro)

    # EXTRACT FEATURES
    df_features = get_features(df)

    # COMPUTE P_VALUE
    base_len    = 10
    compare_len = base_len
    if base_len > len(df_features):
        raise Exception("Dataset too short. Make base len smaller or check data")

    df_metrics = pd.DataFrame(data=df_features[["id","mean"]].iloc[base_len:],columns=["id","mean"])

    p_max= []
    p_min= []
    p_mean = []
    p_list = []
    for i in range(len(df_features)-base_len):

        init = df_features.iloc[:base_len]
        futu = df_features.iloc[1+i:base_len+1+i]
        t,p = ttest_ind(init["mean"], futu["mean"])
        p_mean.append(p)
        t,p = ttest_ind(init["maximum"], futu["maximum"])
        p_max.append(p)
        t,p = ttest_ind(init["minimum"], futu["minimum"])
        p_min.append(p)
        name = "p_"+str(i)
        p_list.append(name)
    df_metrics["name"] = p_list
    df_metrics["p-mean"] = p_mean
    df_metrics["p-max"] = p_max
    df_metrics["p-min"] = p_min
    df_metrics["p"] = df_metrics["p-mean"]
    
    # CHECK NEW DATA
    try:
        max_ptime = get_max_p_value(machine)
    except:
        max_ptime = 0

    values = []
    for p,t in zip(df_metrics["p"].values,times[10:]):
        if float(t) > max_ptime:
            values.append((str(t),str(machine),str(p)))
    
    # INSERT NEW DATA
    if len(values) > 0:
        table_name = "p_value"
        sql = """INSERT INTO """+table_name+""" (TIME,MACHINE,PVALUE) VALUES %s """
        insert_values(table_name,values,sql)
    else:
        print("No values to insert")
        
    return "OK"    
    
def sc_eval_pvalue(**context):

    value = compute_p_value(0)
    value = compute_p_value(1)
    value = compute_p_value(2)
