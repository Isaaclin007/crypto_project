import datetime
import numpy as np
import pandas as pd

import mean_reversion

def Plotting_and_Testing(params, data, plot_obj, test_obj, test_name):
    p_name = params.pair_name
    plot_obj.plot_price_series(data, p_name)
    plot_obj.plot_scatter_series(data, p_name[0], p_name[1])

    # params.ols_res, params.ols_spread = test_obj.OLS_spread()
    params.j_res, params.j_spread = test_obj.Johansen(params.pair_name, 100, True)

    if test_name == "OLS":
        spread = params.ols_spread
    elif test_name == "Johansen":
        spread = params.j_spread

    plot_obj.plot_spread(spread)

    params.cadf_res = test_obj.CADF(spread, True)
    params.Hurst = test_obj.Hurst(spread)

    params.VR = test_obj.Variance_Ratio(spread, 100)
    plot_obj.plot_VR(params.VR)

    params.half_life = test_obj.half_life(spread)
    params.z_score = test_obj.dynamic_z_score(spread, 5, 16, True)

    plot_obj.plotting_z_score(params.z_score)

def main():
    df = pd.read_csv("eth_data.csv", engine = "c", usecols = range(1,10))
    df_last = pd.DataFrame()
    df_last["utc"] = df["utc"].apply(lambda x: datetime.datetime.strptime("20" + x, '%Y-%m-%d %H:%M'))
    df_last["x"], df_last["y"] = df["last.x"], df["last.y"]
    df_last = df_last.set_index("utc")

    params = mean_reversion.params(["x", "y"], datetime.datetime(2018, 10, 1), datetime.datetime(2019, 1, 2))
    plotting = mean_reversion.plotting(params.start_time, params.end_time, df_last.index)
    testing = mean_reversion.Testing(df_last)
    Plotting_and_Testing(params, df_last,plotting, testing, "Johansen")

if __name__ == "__main__":
    main()
