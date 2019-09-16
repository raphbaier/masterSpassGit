import yfinance as yf

data = yf.download('','2018-01-01','2018-01-08')

strings_with_problems = ("BF.B", "BF.2", "DWDP", "HRS")