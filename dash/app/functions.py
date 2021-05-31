import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import flask
from dash.dependencies import (Input, Output)
import requests
import time
import numpy as np
import plotly.graph_objs as go
import psycopg2
from psycopg2.extras import execute_values
import time

def get_data(table_name,sql):
    """ query data from the cell tables table """
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

def connect_postgress():
    """ open conexion to the PostgreSQL database"""
    conn = psycopg2.connect(
        host="postgres-data",
        port=5432,
        database="sc",
        user="postgres",
        password="postgres")
    return conn

def get_last_machine(machine):
    sql = """
    SELECT a.TIME,a.MACHINE,a.DATA 
    FROM profiles a
    INNER JOIN(
       SELECT max(TIME) as TIME,MACHINE
       FROM profiles
       GROUP BY MACHINE
    ) b on a.TIME = b.TIME AND a.MACHINE = b.MACHINE
    """
    values =read_data(sql)
    values = [float(s) for s in values[machine][2][2:-2].split(",")]
    return values


def get_metrics(machine):
    sql = """
    SELECT MACHINE,MEAN,MAX,MIN,TIME FROM profiles ORDER BY TIME
    """
    values =read_data(sql)
    mean = []
    max = []
    min = []
    for item in values:
        if int(item[0]) == machine:
            mean.append(item[1])
            max.append(item[2])
            min.append(item[3])
    return mean,max,min

def get_p(machine):
    sql = """
    SELECT MACHINE,PVALUE,TIME FROM p_value ORDER BY TIME
    """
    values = read_data(sql)
    pvalue = []
    for item in values:
        if int(item[0]) == machine:
            pvalue.append(item[1])
    return pvalue