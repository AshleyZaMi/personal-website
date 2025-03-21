""" Ashley Zapata Minero
    Project04
    Description: This project is about using the avocado.csv dataset and creating a raw graph on left side where data is not yet
    made into valuable data. In that dataset we plot AveragePrice vs Time and Total Volume vs Time, In the middle column we plot aggregated data,
    which is done by grouping AveragePrice and Total Volume by Date and this gives us a groupby object which creates Date as an index and we then have
    to aggregate by summing function that gives us useful data and turns into a dataset and since AveragePrice was incorrectly processed then we had to modify
    that. We then plot this infomration similarily to Raw but now the data is aggregated. In the third plot we use the aggregated dataset but apply .rolling()
    method to make the data more meaningful than before because it smooths out the important data from the dataset in longer-term trends in a particular period.
    We plot the smoothed and make modifications to the graph to represent the data through a forloop.
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime  # importing the datetime package


def datetimeConversion(df_avocado):
    # converting Date column into a datetime series
    df_avocado['Date'] = pd.to_datetime(df_avocado['Date'])  # q: do we have to use the formatting?
    # print(df_avocado.info())
    # print(df_avocado)
    return df_avocado


def rawPlots(ax, df_avocado):
    # Plotting Average Price vs Time
    ax[0, 0].scatter(df_avocado['Date'], df_avocado['AveragePrice'])
    # Plotting Total Volume vs Time
    ax[1, 0].scatter(df_avocado['Date'], df_avocado['Total Volume'])

    # need to aggregate both AveragePrice and Total Volume by Date
    # creating a new column in df_avocado called TotalRevenue
    df_avocado['TotalRevenue'] = df_avocado['AveragePrice'] * df_avocado['Total Volume']
    return df_avocado


def groupbyAndAggregate(df_avocado):
    # group the df_avocado by Date| creating a groupby object
    # **when groupby is conducted it automatically makes Date the index
    date_grouped = df_avocado.groupby('Date')

    # aggregate to produce groups that make a new Dataframe| and make meaningful data
    df_date_groups = date_grouped.sum()
    return df_date_groups


def AvgPriceCorrection(df_date_groups):
    # correcting AveragePrice by dividing TotalRevenue/Total Volume
    df_date_groups['AveragePrice'] = df_date_groups['TotalRevenue'] / df_date_groups['Total Volume']
    # print(df_date_groups)
    return df_date_groups


def aggregatedPlots(ax, df_date_groups):
    # in middle subplots plot the aggregated AveragePrice vs Time(accessed using .index)
    ax[0, 1].scatter(df_date_groups.index, df_date_groups['AveragePrice'])
    ax[0, 1].plot(df_date_groups.index, df_date_groups['AveragePrice'])

    # plot Total Volume vs Time
    ax[1, 1].scatter(df_date_groups.index, df_date_groups['Total Volume'])
    ax[1, 1].plot(df_date_groups.index, df_date_groups['Total Volume'])


def smoothAggregatedPlot(ax, df_date_groups):
    # in the right two supplots:
    # plot smooth aggregated AveragePrice vs Time| will do rolling which is grouping n neighbor records
    ax[0, 2].scatter(df_date_groups.index, df_date_groups['AveragePrice'].rolling(20).mean())
    ax[0, 2].plot(df_date_groups.index, df_date_groups['AveragePrice'].rolling(20).mean())

    # plot aggreagated Total Volume vs Time| similarily will do rolling here
    ax[1, 2].scatter(df_date_groups.index, df_date_groups['Total Volume'].rolling(20).mean())
    ax[1, 2].plot(df_date_groups.index, df_date_groups['Total Volume'].rolling(20).mean())


def main():
    # write your code here
    # generating a 2x3 subplot grid
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)

    # collecting the data
    file_path = "avocado.csv"
    df_avocado = pd.read_csv(file_path)
    # print(df_avocado)

    # Attributes I want:
    # Date, AveragePrice, Total Volume
    df_avocado = df_avocado[['Date', 'AveragePrice', 'Total Volume']]
    # print(df_avocado)
    # print(df_avocado.info())

    # converting Date column into a datetime series
    df_avocado = datetimeConversion(df_avocado)

    # Plotting Average Price vs Time and Total Volume vs Time
    df_avocado = rawPlots(ax, df_avocado)

    # group the df_avocado by Date| creating a groupby object and aggregate to produce groups that make a new Dataframe| and make meaningful data
    df_date_groups = groupbyAndAggregate(df_avocado)

    # correcting AveragePrice by dividing TotalRevenue/Total Volume
    df_date_groups = AvgPriceCorrection(df_date_groups)

    # in middle subplots plotting Aggregated| the aggregated AveragePrice vs Time and plot Total Volume vs Time| Time was accessed with .index
    # not returning the data since no modifications were made to it
    aggregatedPlots(ax, df_date_groups)

    # in right two subplots plotting smooth aggregated AveragePrice vs Time and plotting aggreagated Total Volume vs Time| applying rolling to both that groups n neighbor records
    # not returning the data since no modifications were made to it
    smoothAggregatedPlot(ax, df_date_groups)

    # making sure all indices in plot have the same indices and also for labeling purposes of the chart
    for i in range(2):
        for k in range(3):
            # setting the axis to look like the time is rotated
            ax[i, k].tick_params(axis='x', rotation=90)
            # only if it's in the 0,0 row,column will print Average Price (USD
            if i == 0 and k == 0:
                ax[i, k].set(ylabel='Average Price (USD)')
            # only if it's in the 1,0 row,column will print Total Volume (millions)
            if i == 1 and k == 0:
                ax[i, k].set(ylabel='Total Volume (millions)')
            # if it's in the 0 row and matches those indices will be named that indicated chart
            if i == 0:
                if k == 0:
                    ax[i, k].set(title="Raw")
                if k == 1:
                    ax[i, k].set(title="Aggregated")
                if k == 2:
                    ax[i, k].set(title="Smoothed")
            # if it's in the 1st row will have Time in x axis
            if i == 1:
                ax[i, k].set(xlabel='Time')

    # to get everything in the figure formatting
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Avocado Prices and Volume Time Series')
    fig.savefig('avocado.png')


if __name__ == '__main__':
    main()