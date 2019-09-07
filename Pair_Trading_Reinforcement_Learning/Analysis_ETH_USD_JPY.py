import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import os
from functools import reduce
from statsmodels.tsa.stattools import coint


sns.set(style='white')

# Retrieve intraday price data and combine them into a DataFrame.
# 1. Load downloaded prices from folder into a list of dataframes.
folder_path = 'STATICS/ETH'
file_names  = os.listdir(folder_path)
tickers     = [name.split('.')[0] for name in file_names]
df_list     = [pd.read_csv(os.path.join('STATICS/ETH', name)) for name in file_names]


# 2. Replace the closing price column name by the ticker.
for i in range(len(df_list)):
    df_list[i].rename(columns={'Mid': tickers[i]}, inplace=True)


# 3. Merge all price dataframes. Extract roughly the first 70% data.
df  = reduce(lambda x, y: pd.merge(x, y, on='Time'), df_list)
idx = round(len(df) * 0.7)
df  = df.iloc[:idx, :]


# Calculate and plot price correlations.
pearson_corr  = df[tickers].corr()
sns.clustermap(pearson_corr).fig.suptitle('Pearson Correlations')


# Plot the marginal distributions.
sns.set(style='darkgrid')
sns.jointplot(df['ETH-USD'], df['ETH-JPY'],  kind='hex', color='#2874A6')


# Calculate the p-value of cointegration test for USD-JPY and KO-PEP pairs.
x = df['ETH-USD']
y = df['ETH-JPY']
_, p_value, _ = coint(x, y)
print('The p_value of USD-ETH-JPY pair cointegration is: {}'.format(p_value))

# Plot the linear relationship of the USD-JPY pair.
df2 = df[['ETH-USD', 'ETH-JPY']].copy()
spread = df2['ETH-USD'] - df2['ETH-JPY']
mean_spread = spread.mean()
df2['Dev'] = spread - mean_spread
rnd = np.random.choice(len(df), size=500)
sns.scatterplot(x='ETH-USD', y='ETH-JPY', hue='Dev', linewidth=0.3, alpha=0.8,
                data=df2.iloc[rnd, :]).set_title('ETH-USD-JPY Price Relationship')



# Plot the historical USD-JPY prices and the spreads for a sample period.
def plot_spread(df, ticker1, ticker2, idx, th, stop):

    px1 = df[ticker1].iloc[idx] / df[ticker1].iloc[idx[0]]
    px2 = df[ticker2].iloc[idx] / df[ticker2].iloc[idx[0]]

    sns.set(style='white')

    # Set plotting figure
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

    # Plot the 1st subplot
    sns.lineplot(data=[px1, px2], linewidth=1.2, ax=ax[0])
    ax[0].legend(loc='upper left')

    # Calculate the spread and other thresholds
    spread = df[ticker1].iloc[idx] - df[ticker2].iloc[idx]
    mean_spread = spread.mean()
    sell_th     = mean_spread + th
    buy_th      = mean_spread - th
    sell_stop   = mean_spread + stop
    buy_stop    = mean_spread - stop

    # Plot the 2nd subplot
    sns.lineplot(data=spread, color='#85929E', ax=ax[1], linewidth=1.2)
    ax[1].axhline(sell_th,   color='b', ls='--', linewidth=1, label='sell_th')
    ax[1].axhline(buy_th,    color='r', ls='--', linewidth=1, label='buy_th')
    ax[1].axhline(sell_stop, color='g', ls='--', linewidth=1, label='sell_stop')
    ax[1].axhline(buy_stop,  color='y', ls='--', linewidth=1, label='buy_stop')
    ax[1].fill_between(idx, sell_th, buy_th, facecolors='r', alpha=0.3)
    ax[1].legend(loc='upper left', labels=['Spread', 'sell_th', 'buy_th', 'sell_stop', 'buy_stop'], prop={'size':6.5})

idx = range(11000, 12000)
plot_spread(df, 'ETH-USD', 'ETH-JPY', idx, 0.5, 1)
plt.show()
