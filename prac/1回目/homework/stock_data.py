import yfinance as yf

ticked = yf.Ticker("^NDX")
hist = ticked.history(period="100d",interval="1d")

print(hist) #ここをコメントアウトするとデータの詳細が見れる

day = hist.index.values
closes=hist['Close'].values #詳細データから必要なデータをlistで取り出した

print(day)  #どんなデータか確認してみよう