import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.model_selection import train_test_split

class plotting():
    def __init__(self, start, end, time_index):
        """Initialling Time"""
        self.start = start
        self.end = end
        print("Start Time:", self.start)
        print("End Time:", self.end)
        self.time_index = time_index

    def plot_price_series(self, df, name):
        months = mdates.MonthLocator() # Every month
        fig, ax = plt.subplots(figsize = (15, 10))
        for i in name:
            ax.plot(self.time_index, df[i], label = i)

        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_xlim(self.start, self.end)
        ax.grid(True)
        fig.autofmt_xdate()

        plt.xlabel("Month/Year")
        plt.ylabel("Price ($)")
        plt.title("Price")
        plt.legend(loc = "center left", bbox_to_anchor= (1, 1.0), ncol=2, borderaxespad=0, frameon=False)
        plt.show()

    def plot_scatter_series(self, df, ts1, ts2):
        fig = plt.figure(figsize = (15, 10))
        plt.xlabel("%s price ($)" % ts1)
        plt.ylabel("%s price ($)" % ts2)
        plt.title("%s and %s Price Scatter Plot" % (ts1, ts2))
        plt.scatter(df[ts1], df[ts2], s = 0.1)
        plt.show()

    def plot_spread(self, df):
        months = mdates.MonthLocator() # Every Month
        fig, ax = plt.subplots(figsize = (15, 10))
        ax.plot(self.time_index, df, label = "spread")
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_xlim(self.start, self.end)
        ax.grid(True)
        fig.autofmt_xdate()

        plt.xlabel("Month/Year")
        plt.ylabel("Price ($)")
        plt.title("Residual Plot")
        plt.legend()

        plt.show()

    def spread_qq_plot(self, df):
        fig = sm.qqplot(df, stats.t, distargs = (10, ), loc = 0, scale = 1)
        plt.show()

    def plot_VR(self, data):
        plt.plot(data)
        plt.xlabel("Time Lag")
        plt.ylabel("Varianc Ratio")
        plt.show()

    def plotting_z_score(self, z):
        plt.figure(figsize = (15, 10))
        plt.plot(z)
        plt.axhline(0, color='black')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.axhline(2.0, color='red', linestyle='--')
        plt.axhline(-1.0, color='green', linestyle='--')
        plt.axhline(-2.0, color='green', linestyle='--')
        plt.legend(['Rolling spread z-Score', 'Mean', '+1', '-1'])
        plt.show()

    def plotting_signal(self, df, params):
        spread = df["spread"]
        z = df["z_score"]
        s1 = df.iloc[:, 0]
        s2 = df.iloc[:, 1]
        # Initialling a new series data to show signal
        buy = spread.copy()
        sell = spread.copy()
        # Deleting the date on which z-score deviate from one std
        buy[z < -1] = np.nan
        sell[z > 1] = np.nan
        # The remaining index is the date we can trade #
        fig_1 = plt.figure(figsize = (15, 10))
        plt.plot(spread, linewidth = 1)
        plt.axhline(spread.mean(), color = "red")
        plt.legend(["AAPL_HPQ"])
        plt.scatter(buy.index, buy, color = "g", marker = "^", s = 20)
        plt.scatter(sell.index, sell, color = "r", marker = "^", s = 20)
        x1, x2, y1, y2 = plt.axis()
        # plt.axis((x1,x2, spread_1.min(),spread_1.max()))
        plt.xlabel("Time")
        plt.ylabel("Spread")
        plt.legend(["Ratio", "Mean", "Buy Signal", "Sell Signal"])
        plt.show()

        buy_spread = 0 * s1.copy()
        sell_spread = 0 * s2.copy()

        fig_2 = plt.figure(figsize = (15, 10))
        # Buy the spread --> Buy s1 and sell s2
        buy_spread_signal = buy.dropna(axis = 0).index
        buy_spread[buy_spread_signal] = s1[buy_spread_signal]
        sell_spread[buy_spread_signal] = s2[buy_spread_signal]
        # Sell the spread --> Sell s1 and buy s2
        sell_spread_signal = sell.dropna(axis = 0).index
        buy_spread[sell_spread_signal] = s2[sell_spread_signal]
        sell_spread[sell_spread_signal] = s1[sell_spread_signal]

        plt.plot(s1, color = "skyblue", linewidth = 1)
        plt.plot(s2, color = "gold", linewidth = 1)

        # Labelling signal

        plt.scatter(buy_spread.index, buy_spread, color = "g", linestyle = "None", marker = "^", s = 20)
        plt.scatter(sell_spread.index, sell_spread, color = "r", linestyle = "None", marker = "^", s = 20)

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1,x2,min(s1.min(),s2.min()),max(s1.max(),s2.max())))
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend([params.pair_name[0], params.pair_name[1], "Buy Signal", "Sell Signal"])
        plt.show()

    def help():
        print("plot_price_series: data start with time index and specific start and end time are needed")
        print("\n")
        print("plot_scatter_series: data and name of different asset are required")
        print("\n")
        print("plot_residuals: a time-series of data after combination is needed")
        print("\n")
        print("plotting_z_score: z score is a time-series data")
        print("\n")
        print("plotting_signal: df contain Two Price series, one spread series, and a z_score series")

class params():
    """Encapsulate model parameters"""
    def __init__(self, pair_name, s_time, e_time):
        """Tuple including name of underlyings"""
        self.pair_name = pair_name

        """Setting Start and End Time"""
        self.start_time = s_time
        self.end_time = e_time

        """Results from Linear Regression and Johansen Tests"""
        self.ols_res = None
        self.j_res = None

        """spread is the cointegrated stationary data"""
        self.ols_spread = None
        self.j_spread = None

        """parameters to test cointegration """
        self.cadf_res = None
        self.Hurst = None
        self.VR = None
        self.half_life = None

    def getattr(self):
        return self.pair_name, self.start_time, self.end_time, self.ols_res, self.j_res, self.ols_spread, self.j_spread, self.cadf_res, self.Hurst, self.VR, self.half_life

class Testing():
    def __init__(self, data, name_lyst):
        self.data = data
        self.name_lyst = name_lyst

    def OLS_spread(self):
        x = self.data[self.name_lyst[0]]
        y = self.data[self.name_lyst[1]]
        OLS = sm.OLS(y, x)
        results = OLS.fit()
        ols_spread = y - results.params[0] * x
        return results, ols_spread.values

    def Johansen(self, p, verbose):
        """
            Get the cointegration vectors at 95% level of significance
            given by the trace statistic test.
        """
        y = self.data[self.name_lyst]
        N, l = y.shape
        jres = coint_johansen(y, 0, p)

        tr_stats = pd.DataFrame(jres.lr1, columns = {"Trace Statistic"})
        tr_stats.index.names = ["NULL: r <= "]
        tr_stats["Criti_90%"], tr_stats["Criti_95%"], tr_stats["Criti_99%"] = jres.cvt[:,0], jres.cvt[:,1], jres.cvt[:,2]

        eign_stats = pd.DataFrame(jres.lr2, columns = {"Eigen Statistic"})
        eign_stats.index.names = ["NULL: r <= "]
        eign_stats["Criti_90%"], eign_stats["Criti_95%"], eign_stats["Criti_99%"] = jres.cvm[:,0], jres.cvm[:,1], jres.cvm[:,2]

        eigen = pd.DataFrame(jres.eig, columns = {"Eigen Value"})
        EVEC = pd.DataFrame(jres.evec)

        if verbose == True:
            print(tr_stats, "\n")
            print(eign_stats, "\n")
            print(eigen, "\n")
            print(EVEC)

        jres.trace = (tr_stats["Trace Statistic"] > tr_stats["Criti_95%"])
        jres.eigen = (eign_stats["Eigen Statistic"] > eign_stats["Criti_95%"])

        jres.max_eigen_ix = np.argmax(jres.eig)
        jres.max_evec = EVEC[jres.max_eigen_ix]

        return jres, np.dot(y, jres.max_evec)

    def CADF(self, df, verbose):

        cadf = ts.adfuller(df)

        if verbose == True:
            print("ADF static", cadf[0])
            print("pValue", cadf[1])
            print("Critical Value", cadf[4])

        return cadf
    def Hurst(self, df):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create time lags
        lags = np.arange(2, 100)

        # Calculat the array of the variance of lagged difference
        func = lambda lag: np.sqrt(np.std(np.subtract(df[lag:], df[:-lag])))
        vfunc = np.vectorize(func)
        tau = vfunc(lags)

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0] * 2

    def Variance_Ratio(self, df, max_lag):
        # Create time lags
        lags = np.arange(1, max_lag)

        # Calculate the array of the variance of lagged difference
        func = lambda lag: np.var(np.subtract(df[lag:], df[:-lag]))/lag
        vfunc = np.vectorize(func)

        # Calculate variance of one day lag
        var_1 = np.var(np.subtract(df[1:], df[:-1]))

        # Calculate Variance Ratio
        VR = vfunc(lags)/var_1

        return VR

    def half_life(self, data):
        """We can calculate this by running a linear regression
           between the spread series and a lagged version of itself.
           The Beta coefficient produced by this regression
           can then be incorporated into the
           Ornstein-Uhlenbeck process to calculate the half-life.
        """

        # Requires data of res OR spread
        df = pd.Series(data)
        res_lag = df.shift(1)
        res_lag.iloc[0] = res_lag.iloc[1]
        res_ret = df - res_lag
        res_ret.iloc[0] = res_ret.iloc[1]
        res_lag_2 = sm.add_constant(res_lag)

        # Building regression model between res_ret and res_lag_2
        model = sm.OLS(res_ret, res_lag_2)
        results = model.fit()

        # Calculating half_life
        half_life = -np.log(2)/results.params[0]

        return half_life

    def get_data(self):
        return self.data

    def help(key_word):
        if key_word == "Johansen" or "CADF":
            print("Input data are prices series named with ols_spared or J_spread")

        elif key_word == "Half-life":
            print("Input data are spread stationary data")

        elif key_word == "Hurst":
            print("Input data are spread stationary data")

        elif key_word == "Variance Ratio":
            print("Input data are spread stationary data")

class coint_matrix():
    def __init__(self, df, name_lyst, test_name, test_method):
        self. df = df
        self.name_lyst = name_lyst
        self.test_name = test_name
        self.test_method = test_method

    def coint_pair(self):
        l = df.shape[1]

        pairs = list(itertools.combinations(self.name_lyst, 2))
        coint_pairs = []

        if self.test_name == "unit root":

            score_mat = pd.DataFrame(np.zeros((l, l)), columns = name, index = name)
            p_val_mat = pd.DataFrame(np.ones((l, l)), columns = name, index = name)

            for i in pairs:

                res = self.test_method(df[i[0]], df[i[1]])

                score_mat.loc[i[0], i[1]] = res[0]
                p_val_mat.loc[i[0], i[1]] = res[1]

                if res[1] < 0.01:
                    coint_pairs.append((i, res[0], res[1]))

            return coint_pairs, score_mat, p_val_mat

        elif self.test_name == "Johansen":
            trace_score_mat = pd.DataFrame(np.zeros((l, l)), columns = name, index = name)
            eigen_score_mat = pd.DataFrame(np.zeros((l, l)), columns = name, index = name)

            for i in pairs:

                res = self.test_method(pd.concat((df[i[0]], df[i[1]]), axis = 1))

                # Statistics to reject Hypothesis that r <= 1

                eigen_score_mat.loc[i[0], i[1]] = res.lr2[0]

                trace_score_mat.loc[i[0], i[1]] = res.lr1[0]

                if res.lr2[0] > res.cvm[0,2]:

                    coint_pairs.append((i, res.eig, res.evec))

            return coint_pairs, trace_score_mat, eigen_score_mat

        else: pass
