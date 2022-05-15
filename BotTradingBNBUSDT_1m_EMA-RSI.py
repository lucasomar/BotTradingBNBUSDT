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

from binance.client import Client
    
Api_key = 'C2CEE5GdQ2YbRblHv9irCduOCmgklzRjq7pxKUtyMnLAwKIDBjeX9DhblEls2Hia'
Secret_key = 'Zckvuv9gdz3QOSdHxmv37Q3KcKk3bTujzHcsgDnWt4E274yexfRwSYz6ZYYbHzPI'


client = Client(Api_key, Secret_key)
import yfinance as yf
import pandas as pd, json, numpy as np
import requests, datetime as dt, threading, time, math
from datetime import datetime as dt
import pickle

pd.set_option('display.max_columns', None)

# Funciones
# Levanto el modelo
def traerModelo(tipo='RF'):
    if tipo=='RF':
        with open('bot_rf_v0_1m_EMA-RSI.dat', 'rb') as file:
            modelo = pickle.load(file)
    else: 
        modelo = None
    return modelo

#modelo = traerModelo(tipo='RF')

def downloadBinance(symbol, interval,startTime=None, endTime=None, limit=1000):
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

#data = downloadBinance('BNBUSDT', '1m', limit=1000)

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
    
  
    #data = data.drop(['Open', 'High', 'Low', 'qVolume', 'trades','Close'],axis=1)
    df = df.dropna()
    return df


def traerData(symbol, interval='1m'):
    try:
        data = downloadBinance('BNBUSDT', interval, limit=1000)
        return data
    except:
        print(f'No se pudo traer la última data de {symbol}')
        return None


def predecir(data, modelo):
    try:
        actual = generarIndicadores(data).iloc[-1,4:]
        y_pred = modelo.predict((actual,))[0]
        y_proba = modelo.predict_proba((actual,))[0]
        return y_pred, y_proba
    except:
        print('No se pudo predecir')
        return None, None


# Funciones de Ruteo
def binanceConect():
    from binance.client import Client
    Api_key = 'C2CEE5GdQ2YbRblHv9irCduOCmgklzRjq7pxKUtyMnLAwKIDBjeX9DhblEls2Hia'
    Secret_key = 'Zckvuv9gdz3QOSdHxmv37Q3KcKk3bTujzHcsgDnWt4E274yexfRwSYz6ZYYbHzPI'
    client = Client(Api_key, Secret_key)



def getLastPrice(symbol, limit=500):
    url = 'https://api.binance.com/api/v3/trades'
    params = {'symbol':symbol,'limit':limit}
    r = requests.get(url,params=params)
    js = r.json()
    js = js[-1]
    #df = pd.DataFrame(js)
    return float(js.get('price'))


#timestamp = int(dt.datetime.today().timestamp()*1000)

'-----------------------------------------------------------------------'
# Funciones para obtener cantidades de BNB y USDT en Spot Wallet
def getQuantityUSDT():
    balance = client.get_asset_balance(asset='USDT')
    USDT = {}
    for k in balance.keys():
        if balance[k] is not None:
            USDT[k] = balance[k]
    USDT = float(USDT['free'])
    return USDT

USDT = getQuantityUSDT()

def getQuantityBNB():
    balance = client.get_asset_balance(asset='BNB')
    BNB = {}
    for k in balance.keys():
        if balance[k] is not None:
            BNB[k] = balance[k]
    BNB = float(BNB['free'])
    return BNB

BNB = getQuantityBNB()

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


'-----------------------------------------------------------------------'
# Funciones de compra/venta a mercado
def buy_market():
    client.create_order(symbol=symbol, side='BUY', type='MARKET',quantity = round(USDT/getLastPrice(symbol)*0.95,3))

def sell_market():
    client.create_order(symbol=symbol, side='SELL', type='MARKET',quantity = round(getQuantityBNB()*0.97,3)) 

def getLastTrade():
    trades = client.get_my_trades(symbol=symbol)
    return trades[-1]

def getLastTradeTime():
    trades = client.get_my_trades(symbol=symbol)
    return dt.fromtimestamp((trades[-1]['time'])/1e3)
'-----------------------------------------------------------------------'
hoy = dt.now()
start = dt.strftime(hoy,"%Y-%m-%d")
def derrape():
    vix = yf.download('^VIX', start=start)
    vixchange = ((vix['Adj Close']/vix['Open']-1)*100)[0]
    salida = False
    if vixchange > 3:
        salida = True
    
    return salida

'-----------------------------------------------------------------------'
# Funcion principal BOT
def ejecutar(modeloCompra):
    ahora = dt.now()
    hora_decimal = round(ahora.hour + ahora.minute/60 + ahora.second/3600 + ahora.microsecond/(3.6*10**9) ,5)
    USDT = getQuantityUSDT()
    
    if USDT < 20:
        hora_compra = getLastTradeTime()
        hora_compra_decimal = round(hora_compra.hour + hora_compra.minute/60 + hora_compra.second/3600 + hora_compra.microsecond/(3.6*10**9) ,5)
        delta_tiempo = ahora - hora_compra
        tiempo_tenencia = round((delta_tiempo.seconds+delta_tiempo.microseconds/1000000)/3600,5)
        
        if tiempo_tenencia > 40/60:
            BNB = round(getQuantityBNB()*0.97,3)
            sell_market()
            print('Vendiendo a mercado')
            time.sleep(20)
        else:
            BNB = round(getQuantityBNB()*0.97,3)
            print(f'Esperando para vender {round(BNB*0.97,3)} a las {hora_compra_decimal+20/60} son las {hora_decimal}')
            time.sleep(60*1)
        
    else:
        data = traerData(symbol, interval='1m')   
        prediccion = predecir(data,modeloCompra)
        if prediccion[1][1] >= 0.55:
            USDT = getQuantityUSDT()
            buy_market()
            print(f'Comprando BNB hora {hora_decimal}, proba suba: {prediccion[1][1]:.2%}')
            time.sleep(60*1)
        else:
            USDT = getQuantityUSDT()
            print(f'Proba actual suba: {prediccion[1][1]:.2%} Esperando comprar u$d {USDT}, a las {hora_decimal}')
            time.sleep(60*1)


'-----------------------------------------------------------------------'
# EL DEMONIO

binanceConect()
symbol = 'BNBUSDT'
modeloCompra = traerModelo('RF')
#salida = derrape()

#if salida == False:
while True:
     ahora = dt.now()
     hora_decimal = round(ahora.hour + ahora.minute/60 + ahora.second/3600,5)
     
        
     if ahora:
         ejecutar(modeloCompra)
     else:
         break
#else:
#    print('VIX derrapando')
