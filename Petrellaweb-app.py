# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:37:27 2022

@author: ricky
"""

import streamlit as st
import pandas as pd
import base64   #per scaricare file da streamlit
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import numpy as np


st.title('VISUALIZZARE PREZZI DI CHIUSURA S&P 500')

st.markdown("""
Questa web-app mostra le **S&P 500** (da Wikipedia) e i corrispondenti 
**stock closing price** (anno-mese-giorno).

* **Fonte dei dati:** Wikipedia (link sottostante)
(https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('Selezionare i settori desiderati')

#%% Creazione del dataframe di S&P 500
@st.cache   #serve per salvare temporaneamente i settori scelti ed evitare la ricarica dell'app in loop
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0] #si trasforma in dataframe il file html
    return df
df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - selezione del settore
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Settore', sorted_sector_unique, sorted_sector_unique)

# Filtro dati
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Mostra le compagnie del settore selezionato')
st.write('Dimensione dataframe: ' + str(df_selected_sector.shape[0]) + ' righe e ' + str(df_selected_sector.shape[1]) + ' colonne.')
st.dataframe(df_selected_sector)


#%% Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(         #si usa yfinance per ottenere il closing price
        tickers = list(df_selected_sector[:10].Symbol),
        period = st.sidebar.selectbox('Seleziona periodo', ["1y","3y","5y", "10y","ytd"]),
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )


#per ottenere un df di soli close prices
dati = data.loc[:, (slice(None), 'Close')]   
st.header('Dataframe dei close prices dei settori selezionati')

#stampa sull'app i dati dei close prices
st.write(dati)
def fldwnld(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPClose.csv">Download CSV File</a>'
    return href


st.markdown(fldwnld(dati), unsafe_allow_html=True)

#stampa sull'app la media dei close prices
st.header('Dataframe delle medie aritmetiche dei close prices di S&P500 o dei settori scelti')

media_dati=dati.mean(axis = 1)
dfmedia_dati= pd.DataFrame(media_dati)
m_dati= dfmedia_dati.rename(columns = {0:'Medie Close'})


st.write(m_dati)

def fldwnldmedie(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPmedie_close.csv">Download CSV File</a>'
    return href

st.markdown(fldwnldmedie(media_dati), unsafe_allow_html=True)


st.header('Grafico delle medie aritmetiche dei settori scelti')
st.line_chart(m_dati)

#S&P 500 Indice generale
sp500ind = pd.read_csv('C:/Users/ricky/Lez_spyder/INDEX_US_S&P US_SPX.csv')
df_ind= pd.DataFrame(sp500ind)

#grafico
y1= df_ind.Open
y2= df_ind.High
y3= df_ind.Low
y4= df_ind.Close

df_ind_date_inv= df_ind.iloc[:,0] #per selezionare la colonna delle date

df_ind_date= df_ind_date_inv[::-1] #per invertire la serie di date

x= pd.DatetimeIndex(df_ind_date)

st.set_option('deprecation.showPyplotGlobalUse', False)
# plot lines
st.header('Indice generale S&P 500 a un anno')

fig, ax = plt.subplots()
ax.plot(x, y1, label = "Open", linestyle="-")
ax.plot(x, y2, label = "High", linestyle="-")
ax.plot(x, y3, label = "Low", linestyle="-")
ax.plot(x, y4, label = "Close", linestyle="-")
ax.legend()

plt.xlabel('Date')
plt.ylabel('Prezzi')
plt.title('S&P 500 Index')
plt.show()

st.pyplot(fig)


#stampa sull'app i log-ret dei close prices
st.header('Istogramma dei log return')
st.write('Dataframe dei log return.')
dati_logret = np.log(dati).diff().dropna()
st.write(dati_logret)

def fldwnldlogret(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPlogret.csv">Download CSV File</a>'
    return href


st.markdown(fldwnldlogret(dati_logret), unsafe_allow_html=True)

histogram = plt.figure(figsize=(8,6))
plt.title("Istogramma dei log-return")
sns.histplot(dati_logret, color='red', stat='density')
st.pyplot(histogram)


# matrice di correlazione dei dati
st.header('Matrice di correlazione dei dati')
st.write('La correlazione è calcolata fra i Close Prices dei titoli facenti parte dei settori selezionati.')
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(dati.corr(), annot=True, cmap='Reds', center=1, 
            linewidth=.5, ax=ax)
st.write(fig)
