import numpy as np
import yfinance as yf

class MultipleTimeSeriesCV:
    """
    Generates train-test splits for time series data with a MultiIndex (symbol, date).
    - Respects time order (no leakage)
    - Supports purge period via lookahead
    """

    def __init__(self,
                sn_splits=3,
                train_period_length=126,
                test_period_length=21,
                lookahead=0,
                shuffle=False):
        self.n_splits = sn_splits
        self.train_length = train_period_length
        self.test_length = test_period_length
        self.lookahead = lookahead
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values('date').unique().sort_values()
        total_period = self.train_length + self.lookahead + self.test_length
        max_start = len(unique_dates) - total_period

        for i in range(self.n_splits):
            split_start = i * self.test_length
            train_start = split_start
            train_end = train_start + self.train_length
            test_start = train_end + self.lookahead
            test_end = test_start + self.test_length

            if test_end > len(unique_dates):
                break  # 範囲外を防止

            train_dates = unique_dates[train_start:train_end]
            test_dates = unique_dates[test_start:test_end]

            train_idx = X.index.get_level_values('date').isin(train_dates)
            test_idx = X.index.get_level_values('date').isin(test_dates)

            train_idx = np.where(train_idx)[0]
            test_idx = np.where(test_idx)[0]

            if self.shuffle:
                np.random.shuffle(train_idx)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
ticked = yf.Ticker("^NDX")
hist = ticked.history(period="100d",interval="1d")
print(hist) 


YEAR = 252

train_period_length = 63
test_period_length = 10
n_splits = int(3 * YEAR/test_period_length)
lookahead =1 

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                        test_period_length=test_period_length,
                        lookahead=lookahead,
                        train_period_length=train_period_length)
