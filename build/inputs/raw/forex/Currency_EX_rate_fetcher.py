import pandas as pd
from datetime import datetime

import os

now_utc = datetime.utcnow()

main_URL = "https://www.alphavantage.co/query?"
var = {
        "function": "function=" + "FX_DAILY",
        "from_symbol": "from_symbol=" + "USD",
        "to_symbol": "to_symbol=" + "JPY",
        # "interval": "interval=" + "1min",
        "outputsize": "outputsize=" + "full",
        "datatype": "datatype=" + "csv",
        "apiURL": "apikey=" + "{HT1VSGJMRN19BN4W}"
      }

URL = main_URL +"&" + var["function"] +"&" + var["from_symbol"] +"&" + var["to_symbol"] +"&" +var["outputsize"] +"&" + var["datatype"] +"&" + var["apiURL"]
#
# import urllib
#
# urllib.request.urlopen("")
if __name__ == "__main__":
    data = pd.read_csv(URL)
    file_name = "USD_JPY_rate_20_YEARS.csv"
    if os.path.isfile(file_name):
        print("Creating File ...")
        data.to_csv(file_name)
    else:
        data.to_csv(file_name, mode = "a")
