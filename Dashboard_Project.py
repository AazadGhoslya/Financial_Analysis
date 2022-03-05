#                            "JAIHANUMANJI"
# -*- coding: utf-8 -*- 
"""
Created on Wed Nov  3 11:23:01 2021

@author: aghoslya
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import pandas_datareader.data as web
import yfinance as yf
import mplfinance as mpf


st.set_page_config(layout="wide")

#TAB1
def tab1():
      
    
    # Add dashboard title and description
    st.markdown("<h1 style='text-align: left; color: gray;'>  Summary </h1>", unsafe_allow_html=True)
    col1,col2 = st.columns(2);
    def GetSummary(ticker):
            return si.get_quote_table(ticker)
     # Add a check box
    show_data = col1.checkbox("Show Summary")   
    

    if ticker != '-':
            info = GetSummary(ticker)
            #Converting the dictionary into data frame
            info = pd.DataFrame.from_dict(info, orient='index')
            info[0] = info[0].astype(str)
            if show_data:
                st.dataframe(info, height=1000)
    
    if ticker != '-':
        Select_A = st.selectbox("Select Time", ["1mo","3mo","6mo","ytd","1y","3y","5y","max"])
        data = yf.download(ticker, period=Select_A)
        x = data.index
        y = data["Close"]
        #Setting the plot
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(data.index,data["Close"])
        #filling the graph 
        ax.fill_between(x,y)
        ax.set_xlabel("Time Duration")
        ax.set_ylabel("Close Price")
        # Plotting the Data
        st.pyplot(fig)
        
        
#TAB 2

def tab2():
    
    # Add dashboard title and description
    
    st.markdown("<h1 style='text-align: left; color: gray;'>  Chart </h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    # Add table to show stock data
    @st.cache
    def GetStockData(tickers, start_date, end_date):
        return pd.concat([si.get_data(tick, start_date, end_date) for tick in tickers])
    
     
    Select_Graph = col1.selectbox("Select Graph Type", ["--","Candle" , "Line",])
    Select_Duration = col2.selectbox("Select Time Duration", ["1mo","3mo","6mo","ytd","1y","3y","5y","max"])
    Select_Interval = col3.selectbox("Select Time Interval", ["1d" ,"1mo"])
    if Select_Graph == "Line":    
       if ticker != '-':
           # Getting the data for selected period and interval
           data = yf.download(ticker, period = Select_Duration,interval = Select_Interval)
           # Getting the trading dates only
           data = data[data.Open.notnull()]
           #plotting the data 
           mpf.plot(data, type='line',mav = (50),volume = True,tight_layout = True,show_nontrading=False)
           st.pyplot()
           
           
    elif Select_Graph == "Candle":
         if ticker != '-':
           # Getting the data for selected period and interval
           data = yf.download(ticker, period = Select_Duration,interval = Select_Interval)
           # Getting the trading dates only
           data = data[data.Open.notnull()]
           #plotting the data 
           mpf.plot(data, type = 'candle', mav = (50),volume= True,tight_layout = True,style = 'yahoo',show_nontrading=False)
           plt.style.use('default')
           st.pyplot()   

        
def tab3():
    
    st.markdown("<h1 style='text-align: left; color: gray;'>  Statistics </h1>", unsafe_allow_html=True)
    
    def GetStatsData(ticker):
        return si.get_stats_valuation(ticker)
    if ticker != '-':
            info = GetStatsData(ticker)
            st.header('Valuation Measures')
            st.dataframe(info)
            
    def GetStats(ticker):
        return si.get_stats(ticker)
    if ticker != '-':
            info = GetStats(ticker)
            st.header('Stock Price History')
            # Slicing the data frame to get different tables
            st.dataframe(info[0:7].reset_index(drop=True))
            st.header('Share Statistics')
            st.dataframe(info[7:19].reset_index(drop=True))
            st.header('Dividends & Splits')
            st.dataframe(info[19:29].reset_index(drop=True))
            st.header('Fiscal Year')
            st.dataframe(info[29:31].reset_index(drop=True))
            st.header('Profitability')
            st.dataframe(info[31:33].reset_index(drop=True))
            st.header('Management Effectiveness')
            st.dataframe(info[33:35].reset_index(drop=True))
            st.header('Income Statement')
            st.dataframe(info[35:43].reset_index(drop=True))
            st.header('Balance Sheet')
            st.dataframe(info[35:49].reset_index(drop=True))
            st.header('Cash Flow Statement')
            st.dataframe(info[49:].reset_index(drop=True))
            
      

def tab4():
    st.markdown("<h1 style='text-align: left; color: gray;'>  Financials </h1>", unsafe_allow_html=True)
   
    
    col1, col2 = st.columns(2)
    Select_Type = col1.selectbox("Select Finance Type", ['--','Income Statement', 'Balance Sheet','Cash Flow'])
    
    Select_Year= col2.selectbox("Select Duration", ['Annual', 'Quarterly'])
    
    def Income(ticker):
        if ticker != '-':
            #Selecting the duration 
            if Select_Year == 'Annual':
            #Getting the income data 
                info = si.get_income_statement(ticker, yearly=True)
            else:
                info = si.get_income_statement(ticker, yearly= False)
            st.dataframe(info)
            
    
    def Balance(ticker):
        if ticker != '-':
            if Select_Year == 'Annual':
                info = si.get_balance_sheet(ticker, yearly=True)
            else:
                info = si.get_balance_sheet(ticker, yearly=False)
            st.dataframe(info)
            
    
    
    def Cash(ticker):
        if ticker != '-':
            if Select_Year == 'Annual':
                info = si.get_cash_flow(ticker, yearly=True)
            else:
                info = si.get_cash_flow(ticker, yearly=False)
            st.dataframe(info)
            
    
    
    if Select_Type == 'Income Statement':
        # Run Income
        Income(ticker)
    elif Select_Type == 'Balance Sheet':
        # Run Balance
        Balance(ticker)
    elif Select_Type == 'Cash Flow':
        # Run Cash
        Cash(ticker)
    
    
    
   
def tab5():
        
        st.markdown("<h1 style='text-align: left; color: gray;'>  Analysis </h1>", unsafe_allow_html=True)
        def GetAnalystInfo(ticker):
            # Getting the analysis info from YFinance
            return si.get_analysts_info(ticker)
        
          
        if ticker != '-':
            info = GetAnalystInfo(ticker)
            #Coverting the data into Data Frames 
            EE = pd.DataFrame(info['Earnings Estimate'])
            Rev = pd.DataFrame(info['Revenue Estimate'])
            EH = pd.DataFrame(info['Earnings History'])
            EPST = pd.DataFrame(info['EPS Trend'])
            EPSR = pd.DataFrame(info['EPS Revisions'])
            GW = pd.DataFrame(info['Growth Estimates'])
            
            
            #Setting the headers for each table
            st.header('Earnings Estimate')
            #Getting the data frame
            st.dataframe(EE)
            st.header('Revenue Estimate')
            st.dataframe(Rev)
            st.header('Earnings History')
            st.dataframe(EH)
            st.header('EPS Trend')
            st.dataframe(EPST)
            st.header('EPS Revisions')
            st.dataframe(EPSR)
            st.header('Growth Estimates')
            st.dataframe(GW)
                 
            
    
def tab6():
    
    st.markdown("<h1 style='text-align: left; color: gray;'> Monte Carlo Simulation </h1>", unsafe_allow_html=True)
    
    col1 , col2 = st.columns(2)
    Select_Simulation = col1.selectbox("Select Simulations", [200, 500,1000])
    Select_Times = col2.selectbox("Select Time", [30, 60, 90])
    
    stock_price = web.DataReader(ticker, 'yahoo', start_date, end_date)
    close_price = stock_price['Close']
    
# The returns ((today price - yesterday price) / yesterday price)
    daily_return = close_price.pct_change()

# The volatility (high value, high risk)
    daily_volatility = np.std(daily_return)
    
    
    # Take the last close price
    last_price = close_price[-1]

# Generate the stock price of next 30 days
    #time_horizon = Select_Times
    next_price = []

    for n in range(Select_Times):
    
    # Generate the random percentage change around the mean (0) and std (daily_volatility)
        future_return = np.random.normal(0, daily_volatility)
    
    # Generate the random future price
        future_price = last_price * (1 + future_return)
    
    # Save the price and go next
        next_price.append(future_price)
        last_price = future_price

# Setup the Monte Carlo simulation
    np.random.seed(123)
    simulations = Select_Simulation
    time_horizone = Select_Times

# Run the simulation
    simulation_df = pd.DataFrame()

    for i in range(simulations):
    
    # The list to store the next stock price
        next_price = []
    
    # Create the next stock price
        last_price = close_price[-1]
    
        for j in range(time_horizone):
        # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)

        # Generate the random future price
            future_price = last_price * (1 + future_return)

        # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
            
        simulation_df[i] = next_price
        
    #print(simulation_df.shape)
    # Plot the simulation stock price in the future
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)

    plt.plot(simulation_df)
    plt.title('Monte Carlo simulation')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.axhline(y=close_price[-1], color='red')
    plt.legend(['Current stock price is:' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
def tab7():
    def get_ticker(name):
        tick = yf.Ticker(name) 
        return tick

    Select_time = st.selectbox("Select Time", ["1mo","3mo","6mo","ytd","1y","3y","5y","max"])
    Data = get_ticker(ticker)

    # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    data1 = Data.history(period=Select_time)
    
    st.write(ticker)
    # Getting the Business Summary
    st.write(Data.info['longBusinessSummary']) 
    
    # plots the graph
    st.line_chart(data1.values)
    
    
# Main Body
def run():
    st.markdown("<h1 style='text-align: center; color: navy;'>YAHOO FINANCE !</h1>", unsafe_allow_html=True)

    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add selection box
    global ticker
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,1])
    ticker = col1.selectbox("Select a ticker", ticker_list)
    
    
    # Add select begin-end date
    global start_date, end_date
    
    start_date = col3.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col4.date_input("End date", datetime.today().date())
    
    # Add a radio box
    select_tab = col2.selectbox("Select Option", ['Summary Dashboard', 'Chart','Statistics','Financials','Analysis','Monte Carlo Simulation','History'])
    
    with col5:
        with st.form(key = "Refresh Form"):
            st.form_submit_button(label = "Refresh")
    
    # Show the selected tab
    if select_tab == 'Summary Dashboard':
        # Run tab 1
        tab1()
    elif select_tab == 'Chart':
        # Run tab 2
        tab2()
    elif select_tab == 'Statistics':
        # Run tab 3
        tab3()
    elif select_tab == 'Financials':
        # Run tab 4
        tab4()
    elif select_tab == 'Analysis':
        # Run tab 5
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        # Run tab 6
        tab6()
    elif select_tab == 'History':
        # Run tab 6
        tab7()
    
    
if __name__ == "__main__":
    run()
    