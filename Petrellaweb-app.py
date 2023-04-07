# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 20:37:27 2022

@author: ricky
"""

import streamlit as st
import pandas as pd
import base64   #to download the file from streamlit
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import numpy as np


st.title('VISUALIZE CLOSE PRICES FROM S&P 500')

st.markdown("""
This web-app shows a dataset from **S&P 500** (downloaded from Wikipedia) 
and the relateed  **stock closing price** (year-month-day).

* **Data Source:** Wikipedia (link below)
(https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('Choose the desired sectors')

#%% creation of the dataframe of S&P 500
@st.cache   #necessary to temporarily save the chosen sectors and to avoid 
# a continuos reboot of the web-app
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0] #transform the html file into a dataframe
    return df
df = load_data()
sector = df.groupby('GICS Sector')

# Sidebar - sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Data filter
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Shows the companies of the selected sector')
st.write('Dataframe dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
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

data = yf.download(         #we get the closing prices from yahoo finance
        tickers = list(df_selected_sector[:10].Symbol),
        period = st.sidebar.selectbox('Select the period', ["1y","3y","5y", "10y","ytd"]),
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )


#to get a df with only close prices
dati = data.loc[:, (slice(None), 'Close')]   
st.header('Dataframe of close prices of selected sectors')

# to show on the app the close prices
st.write(dati)
def fldwnld(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPClose.csv">Download CSV File</a>'
    return href


st.markdown(fldwnld(dati), unsafe_allow_html=True)

#to show on the app the mean of close prices
st.header('Dataframe of arithmetic means of the close prices of the full S&P500 or of the selected sectors')

media_dati=dati.mean(axis = 1)
dfmedia_dati= pd.DataFrame(media_dati)
m_dati= dfmedia_dati.rename(columns = {0:'Close means'})


st.write(m_dati)

def fldwnldmedie(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPmedie_close.csv">Download CSV File</a>'
    return href

st.markdown(fldwnldmedie(media_dati), unsafe_allow_html=True)


st.header('Plot of arithmetical means of the selected sectors')
st.line_chart(m_dati)

#S&P 500 general index
sp500ind = pd.read_csv('C:/Users/ricky/Lez_spyder/INDEX_US_S&P US_SPX.csv')
df_ind= pd.DataFrame(sp500ind)

#plots
y1= df_ind.Open
y2= df_ind.High
y3= df_ind.Low
y4= df_ind.Close

df_ind_date_inv= df_ind.iloc[:,0] #to select the column of the dates

df_ind_date= df_ind_date_inv[::-1] #to invert the list of dates

x= pd.DatetimeIndex(df_ind_date)

st.set_option('deprecation.showPyplotGlobalUse', False)
# plot lines
st.header('S&P 500 1 year general index')

fig, ax = plt.subplots()
ax.plot(x, y1, label = "Open", linestyle="-")
ax.plot(x, y2, label = "High", linestyle="-")
ax.plot(x, y3, label = "Low", linestyle="-")
ax.plot(x, y4, label = "Close", linestyle="-")
ax.legend()

plt.xlabel('Dates')
plt.ylabel('Prices')
plt.title('S&P 500 Index')
plt.show()

st.pyplot(fig)


#to show on the app the log-ret of the close prices
st.header('histograms of log return')
st.write('Dataframe of log return.')
dati_logret = np.log(dati).diff().dropna()
st.write(dati_logret)

def fldwnldlogret(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SPlogret.csv">Download CSV File</a>'
    return href


st.markdown(fldwnldlogret(dati_logret), unsafe_allow_html=True)

histogram = plt.figure(figsize=(8,6))
plt.title("Histogram of log-return")
sns.histplot(dati_logret, color='red', stat='density')
st.pyplot(histogram)


# correlation matrix of the data
st.header('Correlation matrix of the data')
st.write('The correlation is computed between the Close Prices of the stocks from the selected sectors.')
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(dati.corr(), annot=True, cmap='Reds', center=1, 
            linewidth=.5, ax=ax)
st.write(fig)
