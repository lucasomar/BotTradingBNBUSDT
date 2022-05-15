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
import requests, tqdm, sys, os, sqlite3, datetime as dt, threading, time
import pickle
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Funciones

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


# Lo traigo con pickle
with open('BNBUSDT_1m_HistDataV1.pickle', 'rb') as file:
    data = pickle.load(file)
pd.set_option('display.max_columns', None)
# Generamos predictores para Machine Learning

#--------------------------------------------
# Armo los predictores:
def generarIndicadores(data):  
    medias = ((10,20),(20,50),(50,100),(100,150))
    mediasExp = ((3,8),(8,16),(16,32),(32,64),(64,128),(128,256),(256,512))
    ventana = 40
    
    #---Cal_RSI---#
    dif = data['Close'].diff()
    win = pd.DataFrame(np.where(dif > 0, dif, 0))
    loss = pd.DataFrame(np.where(dif < 0, abs(dif), 0))
    ema_win = win.ewm(alpha=1/14).mean()
    ema_loss = loss.ewm(alpha=1/14).mean()
    rs = ema_win / ema_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.index = data.index
    
    
    df = (data['Open']+data['High']+data['Low']+data['Close'])/4
    df = df.to_frame().apply(pd.to_numeric)
    df.columns = ['px']
    
    df['pctChange'] = data['Close'].pct_change()
    df['fw'] = (df.px.shift(-ventana)/df.px -1)*100
    df['rsi'] = rsi/100
    df['roll_vol'] = df['pctChange'].rolling(ventana).std()*ventana**0.5
    df['ema_vol'] = df['pctChange'].ewm(ventana).std()*ventana**0.5
    
    
    df['cruce_1'] = (df['px'].rolling(medias[0][0]).mean() / df['px'].rolling(medias[0][1]).mean() -1 ) *100
    df['cruce_2'] = (df['px'].rolling(medias[1][0]).mean() / df['px'].rolling(medias[1][1]).mean() -1 ) *100
    df['cruce_3'] = (df['px'].rolling(medias[2][0]).mean() / df['px'].rolling(medias[2][1]).mean() -1 ) *100
    df['cruce_4'] = (df['px'].rolling(medias[3][0]).mean() / df['px'].rolling(medias[3][1]).mean() -1 ) *100
    
    
    df['cruce_1exp'] = (df['px'].ewm(span=mediasExp[0][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[0][1], adjust=False).mean() -1 ) *100
    df['cruce_2exp'] = (df['px'].ewm(span=mediasExp[1][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[1][1], adjust=False).mean() -1 ) *100
    df['cruce_3exp'] = (df['px'].ewm(span=mediasExp[2][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[2][1], adjust=False).mean() -1 ) *100
    df['cruce_4exp'] = (df['px'].ewm(span=mediasExp[3][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[3][1], adjust=False).mean() -1 ) *100
    df['cruce_5exp'] = (df['px'].ewm(span=mediasExp[4][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[4][1], adjust=False).mean() -1 ) *100
    df['cruce_6exp'] = (df['px'].ewm(span=mediasExp[5][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[5][1], adjust=False).mean() -1 ) *100
    df['cruce_7exp'] = (df['px'].ewm(span=mediasExp[6][0], adjust=False).mean() / df['px'].ewm(span=mediasExp[6][1], adjust=False).mean() -1 ) *100
    
  
    df = df.dropna()
    return df


df = generarIndicadores(data)


valor_critico_fw = df.fw.quantile(0.6)
print(f'El % crítico es {valor_critico_fw}')

# % medio de las que gana
res_medio_win = df.loc[df.fw > valor_critico_fw].fw.mean()
print(f'La media % win es {res_medio_win}')


# & medio de las que pierde
res_medio_loss = df.loc[df.fw <= valor_critico_fw].fw.mean()
print(f'La media % loss es {res_medio_loss}')


df['pred'] = np.where(df.fw > valor_critico_fw ,1 ,0)
print(df.groupby('pred').size())
print(df.groupby('pred').size() / len(df))

'LLegar a un nivel de 30 por nodo'
max_depth=12
DatosXnodo = len(df)/(2**max_depth)
print(f'Datos por nodo: {DatosXnodo}')

# Entrenamos un modelo con el 60% de los datos (test_size=0.4)
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,4:-1], df.pred, test_size=0.4)

modelo_rf = RandomForestClassifier(criterion = 'entropy', max_depth=max_depth)
modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)
with open('bot_rf_v0_1m_EMA-RSI.dat', 'wb') as file:
    pickle.dump(modelo_rf,file)

m = np.array(skm.confusion_matrix(y_test, y_pred, normalize='all'))
skm.plot_confusion_matrix(modelo_rf, X_test, y_test,values_format='.1%', normalize='all', cmap='Blues')


# Overfiteamos
#aciertos = []
#for i in range(1,25):
#    modelo = RandomForestClassifier(criterion = 'entropy', max_depth=i) #Definimos modelo
#    modelo.fit(X_train, y_train) # Entrenamos
#    y_pred = modelo.predict(X_test) # Corremos
#    m = np.array(skm.confusion_matrix(y_test, y_pred)) # medimos
#    mp = (m/m.sum().sum() *100).round(2)
#    aciertos.append( round(mp[0][0]+mp[1][1],2) )

#aciertosTot = pd.DataFrame(aciertos)
#aciertosTot.plot()
