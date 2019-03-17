from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class trading_criterion():
    def __init__(self, pair_name, window_1, window_2, entry_z, exit_z):
        """Tuple including name of underlyings """
        self.pair_name = pair_name
        """Rolling window for recent mean """
        self.window_1 = window_1
        """Rolling window for lookback period """
        self.window_2 = window_2
        """Entry z-score only one value, + for short and - for long """
        self.entry_z = entry_z
        """Exit z-score to clear position """
        self.exit_z = exit_z

    def getattr(self):
        return self.pair_name, self.window_1, self.window_2, self.entry_z, self.exit_z

class trading_data(trading_criterion):
    def __init__(self, data, pair_name, window_1, window_2, entry_z, exit_z):
        self.data = data
        super(trading_data, self).__init__(pair_name, window_1, window_2, entry_z, exit_z) # Super() function is used for multiple Inheritance

    def process_data(self, beta, j_res, method):
        """beta, j_res, and method are external variable need to be assigned"""
        spread = self.new_spread(self.data, self.pair_name, beta, j_res, method)
        z = self.dynamic_z_score(spread, self.window_1, self.window_2, False)
        df = pd.concat((self.data, spread, z), axis = 1).dropna(axis = 0)
        df["signal"] = df["z_score"].apply(lambda x: self.Signal(x, self.entry_z, self.exit_z))

        return df

    def new_spread(self, df, pair_name, beta, j_res, method):

        df_pair = df[[pair_name[0], pair_name[1]]]
        coint_data = pd.concat((df[pair_name[0]], df[pair_name[1]]), axis = 1)

        if method == "Johansen":
            return pd.DataFrame(np.dot(coint_data, j_res.max_evec), index = df.index, columns = ["J_Spread"])
        elif method == "OLS":
            return pd.DataFrame(coint_data.iloc[:, 1] - beta * coint_data.iloc[:, 0], index = df.index, columns = ["OLS_Spread"])
        else:
            raise Exception("Method Not Assigned")

    def dynamic_z_score(self, df, window_1, window_2, verbose):
        mean_1 = df.rolling(window = window_1, center = False).mean()
        mean_2 = df.rolling(window = window_2, center = False).mean()
        std = df.rolling(window = window_2, center = False).std()

        if verbose == True:
            plt.figure(figsize = (15, 10))
            plt.plot(df.index, df)
            plt.plot(mean_1.index, mean_1)
            plt.plot(mean_2.index, mean_2)

            plt.legend(["spread", str(window_1) + "d mean spread", str(window_2) + "d mean spread"])

            plt.ylabel("spread")
            plt.xlabel("Time")

        z = (mean_1 - mean_2)/std
        return z.mask(np.isinf(z)).rename(columns = {z.columns[0]: "z_score"})

    def Signal(self, x, entry, exit):
        if x > entry:
            return 1
        elif x < -entry:
            return -1
        elif x > -exit and x < exit:
            return 0
        else:
            raise Exception("Score Not Assigned")

class trading_params():
    """This is a parameters class to contain all the information in trading"""
    def __init__(self, data, pair_name, trading_var):
        """
           Initial data should be a Pandas DataFrame
           cointain price, spread, z_score, and signal
           with Time series Index
           No NaN or Inf values are acceptable
        """
        """ Initial fund for trading """
        self.init_f = trading_var[0]
        """ Unit amount for trading - Scalling factor """
        self.unit_amount = trading_var[1]
        """Clearing Time is the timestamp to exit market"""
        self.clear_time = trading_var[2]
        self.data = data
        # Checking NaN and Inf Value in DataFrame
        assert not self.data.isin([np.nan, np.inf, -np.inf]).any().any(), "Nan and Inf detected"
        self.index = self.data.index
        self.z = self.data["z_score"]
        self.signal = self.data["signal"]
        self.asset_0 = self.data[["last_0", "buy_0", "sell_0"]]
        self.asset_1 = self.data[["last_1", "buy_1", "sell_1"]]

        l = len(self.data)
        self.bal = np.zeros((len(data), len(pair_name)))
        self.count = np.zeros((len(data), len(pair_name)))
        self.vol = np.zeros((len(data), len(pair_name)))
        self.cost = np.zeros((len(data), len(pair_name)))

    def getattr(self):
        return self.data, self.bal, self.count, self.vol, self.cost

    def help(self):
        print("Pandas DataFrame TimeIndex, last_0, buy_0, sell_0, last_1, buy_1, sell_1, Spread, z_score, Signal, No NaN or Inf values are acceptable")

class trading_test(trading_params):
    def __init__(self, hedge_ratio, data, pair_name, trading_var):
        super(trading_test, self).__init__(data, pair_name, trading_var) # Only for single Inheritance
        self.hedge_ratio = hedge_ratio

    def back_test(self, rate_lyst, verbose):
        l = len(self.data)
        clear_key = 0
        for i in range(l):
            scale = abs(self.z.iloc[i])
            # Short spread if z > 1:
            if self.signal.iloc[i] == 1:
                self.count[i] = -self.counting(self.count[i], self.hedge_ratio, scale)
                self.bal[i] = self.balancing(self.bal[i], self.count[i], [self.asset_0.iloc[i, -2:], self.asset_1.iloc[i, -2:]])
                self.cost[i] = self.fees(self.cost[i], self.bal[i], rate_lyst)
                self.vol[i] = self.count[i]
                if verbose == True:
                    print(self.index[i], "Sell Spread", self.bal[i], self.vol[i], self.cost[i])
                else: pass
            # Long spread if z < -1:
            elif self.signal.iloc[i] == -1:
                self.count[i] = self.counting(self.count[i], self.hedge_ratio, scale)
                self.bal[i] = self.balancing(self.bal[i], self.count[i], [self.asset_0.iloc[i, -2:], self.asset_1.iloc[i, -2:]])
                self.cost[i] = self.fees(self.cost[i], self.bal[i], rate_lyst)
                self.vol[i] = self.count[i]
                if verbose == True:
                    print(self.index[i], "Buy Spread", self.bal[i], self.vol[i], self.cost[i])
                else: pass
            # Clear position
            elif self.signal.iloc[i] == 0:
                self.count[i,:] = -self.count[clear_key:i, :].sum(axis = 0)
                self.bal[i] = self.balancing(self.bal[i], self.count[i], [self.asset_0.iloc[i, -2:], self.asset_1.iloc[i, -2:]])
                self.cost[i] = self.fees(self.cost[i], self.bal[i], rate_lyst)
                self.vol[i] = self.count[i]
                self.count[clear_key:i+1, :] = 0
                clear_key = i
                if verbose == True:
                    print(self.index[i], "Clear Out", self.bal[i], self.vol[i], self.cost[i], self.index[clear_key])
                else: pass
            # Forced to exit market every certain minutes
            lag = self.clear_time
            if i > lag and self.count[clear_key:i-lag, :].sum(axis = 0)[0] != 0:
                count = -self.count[:i-lag, :].sum(axis = 0)
                bal = self.balancing(self.bal[i].tolist(), count, [self.asset_0.iloc[i, -2:], self.asset_1.iloc[i, -2:]])
                self.bal[i] += bal

                self.cost[i] += self.fees(self.cost[i], bal, rate_lyst)
                self.vol[i] += count
                self.count[clear_key:i-lag+1, :] = 0
                clear_key = i
                if verbose == True:
                    print(self.index[i], "Forced Clearing out", self.bal[i], self.vol[i], self.cost[i], self.index[clear_key])
                else: pass
            else: pass

        col_name = ["Volume_0", "Volume_1", "Balance_0", "Balance_1", "Cost_0", "Cost_1", "Count_0", "Count_1"]
        st_results = pd.DataFrame(np.concatenate((self.vol, self.bal, self.cost, self.count), axis = 1), index = self.data.index, columns = col_name)
        st_results["Net_0"] = st_results["Balance_0"] - st_results["Cost_0"]
        st_results["Net_1"] = st_results["Balance_1"] - st_results["Cost_1"]
        results = st_results.cumsum()

        st_results["Single Net"] = st_results["Balance_0"]+st_results["Balance_1"]-(st_results["Cost_0"]+st_results["Cost_1"])
        results["Net"] = results["Balance_0"]+results["Balance_1"]-(results["Cost_0"]+results["Cost_1"])
        return st_results, results

    def counting(self, c_lyst, hedge_ratio, scale):
        assert not len(c_lyst) != len(hedge_ratio), "Length Mismatch"
        for i in range(len(c_lyst)):
            c_lyst[i] = hedge_ratio[i] #* self.unit_amount * scale

        return c_lyst

    def balancing(self, b_lyst, c_lyst, asset_lyst):
        assert not len(c_lyst) != len(asset_lyst), "Length Mismatch"
        for i in range(len(c_lyst)):
            if c_lyst[i] > 0: # Buying Asset
                b_lyst[i] = -c_lyst[i] * asset_lyst[i]["buy_" + str(i)]
            elif c_lyst[i] < 0: # Selling Asset
                b_lyst[i] = -c_lyst[i] * asset_lyst[i]["sell_" + str(i)]
            elif c_lyst[i] == 0.0:
                b_lyst[i] = 0
            else: pass

        return b_lyst

    def fees(self, cost_lyst, b_lyst, rate_lyst):
        assert not len(cost_lyst) != len(b_lyst), "Length Mismatch"
        for i in range(len(b_lyst)):
            cost_lyst[i] = b_lyst[i] * rate_lyst[i]

        return cost_lyst

    def getattr(self):
        return self.hedge_ratio, self.data, self.bal, self.count, self.vol, self.cost

if __name__ == "__main__":
    df = pd.read_csv("Test_data.csv", engine = "c").set_index("utc")
    df = df[:100]
    trading = trading_test([0.5309, -0.5318], df, ["last_0", "last_1"], [0, 1, 5])
    st_results, results = trading.back_test([0, 0.0025], True)

    import matplotlib.dates as mdates
    months = mdates.MonthLocator() # Every month
    fig, ax = plt.subplots(figsize = (15, 10))
    # plt.plot(results["Net"])
    # plt.plot(results["Balance_0"])
    # plt.plot(results["Balance_1"])
    plt.plot(results["Count_0"])
    plt.plot(results["Count_1"])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.grid(True)
    fig.autofmt_xdate()
    plt.show()
