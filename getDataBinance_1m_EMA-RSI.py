# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:52:13 2021

@author: Lucas
"""

# Bot trading scalping BNBUSDT - Binance

'Características'
# tipo_operacion = swing
# instrumento = BNBUSDT
# timeframe = 1min
# SeñalCompra = modeloML
# SeñalVenta = timeout
# TipoCompraVenta = market (agresivo)
# Driver = Binance
# Ruteo = Binance

'Acciones'
# Lectura de datos API Binance
# Operatoria mercado crypto API Binance
# Predicción --> ModeloML
# Decisión de venta --> Lectura de relor y BBDD con registro de compra
# Persistencia --> A BBDD ????


# Importo librerías y claves Binance API MilanesaGolpeadito:
Api_key = 'C2CEE5GdQ2YbRblHv9irCduOCmgklzRjq7pxKUtyMnLAwKIDBjeX9DhblEls2Hia'
Secret_key = 'Zckvuv9gdz3QOSdHxmv37Q3KcKk3bTujzHcsgDnWt4E274yexfRwSYz6ZYYbHzPI'
import pandas as pd, json, numpy as np, matplotlib.pyplot as plt
import requests, tqdm, sys, os, sqlite3, datetime as dt, threading, time, pickle

def historico(symbol, interval,startTime=None, endTime=None, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol':symbol,
              'interval':interval,
              'startTime':startTime,
              'endTime':endTime,
              'limit':limit}
    r = requests.get(url,params=params)
    js = r.json()
    
    Cols = ['openTime','Open','High','Low','Close','Volume','cTime',
            'qVolume','trades','takerBase','takerQuote','Ignore']
    df = pd.DataFrame(js, columns=Cols)
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.openTime, unit='ms')
    df = df.drop(['openTime','cTime','takerBase','takerQuote','Ignore'],axis=1)
    return df

print(historico('BNBUSDT','1m'))

def listaFechas(fecha_i, fecha_f, minutos_int=1):
    fechas = []
    while fecha_i < fecha_f:
        ts = int(dt.datetime.timestamp(fecha_i))*1000
        fechas.append(ts)
        fecha_i += dt.timedelta(seconds=60*minutos_int*1000)
    return fechas

fecha_i = dt.datetime(2022,1,1)
fecha_f = dt.datetime(2022,3,28)

n_threads = 10
fechas = listaFechas(fecha_i, fecha_f, minutos_int=1)
subs = np.array_split(fechas, n_threads)
print(subs)

dfs = []
def worker(fechas):
    for ts in fechas:
        df = historico('BNBUSDT','1m',startTime=ts)
        dfs.append(df)
    return df

t0 = time.time
threads = []
for i in range(n_threads):
    t = threading.Thread(target=worker, args=(subs[i],))
    threads.append(t)
    t.start()
    
for t in threads:
    t.join()
    
tabla = pd.concat(dfs)


tabla = pd.DataFrame()
for fecha in fechas:
    df = historico('BNBUSDT','1m',startTime=fecha)
    tabla = pd.concat([tabla,df])

tabla.to_csv('BNBUSDT_1m_HistDataV1.csv')


data = pd.read_csv('BNBUSDT_1m_HistDataV1.csv', index_col='openTime')

# Lo guardo
with open('BNBUSDT_1m_HistDataV1.pickle', 'wb') as file:
    pickle.dump(data,file)















