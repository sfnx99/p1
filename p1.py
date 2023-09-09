import csv
import warnings
import time
from datetime import datetime
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

# src https://www.wsj.com/market-data/quotes/fx/USDKRW/historical-prices

MAX_TPE_P = 18
MAX_TPE_D = 2
MAX_TPE_Q = 2
MAX_TPE_EVALS = 30

#------------------------------------------------------------------------------
# 1 - Open Data
#------------------------------------------------------------------------------

filename = 'usdkrw.csv'
dateFormat = '%m/%d/%y,' 

class Price(IntEnum):
    OPEN = 1
    HIGH = 2
    LOW = 3
    CLOSE = 4

rowCount = 3455

data = [[0] * 2 for _ in range(rowCount)]

with open(filename, newline='') as csvfile:
    i = 0
    r = csv.reader(csvfile, delimiter=' ', quotechar='|')
    next(r)
    for row in r:
        data[i][0] = datetime.strptime(row[0], dateFormat)
        data[i][1] = float(row[Price.CLOSE])

        i += 1

#------------------------------------------------------------------------------
# 2 - Process data
#------------------------------------------------------------------------------

df = pd.DataFrame({'date': np.array(data)[:,0],
                            'price': np.array(data)[:,1]})
df['price'] = df['price'].astype(float)

df.set_index('date', inplace=True)

prices = np.array(df['price'])

prices = np.flip(prices)

diffs = list()

for i in range(len(prices) - 1):
    diffs.append(prices[i + 1] - prices[i])

size = int(len(diffs) * 0.8)

train, test = diffs[0:size], diffs[size:len(prices)]

history = [x for x in train]

# plt.plot(df.index[:-1], diffs)
# plt.show()

# autocorrelation_plot(diffs)
# pyplot.show()

#------------------------------------------------------------------------------
# 3 - ARIMA
#------------------------------------------------------------------------------

step = 10
train_proportion = 0.8

#------------------------------------------------------------------------------
# 3.1 - evaluate an ARIMA model for a given order (p,d,q)
#------------------------------------------------------------------------------
def evaluate_arima_model(X, arima_order):

    train_size = int(len(X) * train_proportion)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    # make predictions
    predictions = list()

    for t in range(int(len(test) / step)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(step)
        predictions.extend(yhat)
        history.extend(test[t * step:(t + 1) * step])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test[:-(len(test) % step)], predictions))
    return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(data, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(data, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# 3.2 - evaluate parameters using Grid Search
#------------------------------------------------------------------------------

# p_values = [0, 1, 2, 4, 6, 8, 10]
# d_values = range(0, 3)
# q_values = range(0, 3)

# evaluate_models(diffs, p_values, d_values, q_values)

# model = ARIMA(np.array(df['price']), order=(5,1,0))
# model_fit = model.fit()

# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# # density plot of residuals
# residuals.plot(kind='kde')
# pyplot.show()
# # summary stats of residuals
# print(residuals.describe())

#------------------------------------------------------------------------------
# 3.3 - evaluate parameters using Bayesian Hyperparameter Search
#------------------------------------------------------------------------------

def arima_tpe(params):
    return {"loss": evaluate_arima_model(diffs, params),
        "status": STATUS_OK,
        "eval_time": time.time()}

space = (
    hp.quniform("p", 0, MAX_TPE_P, 1),
    hp.quniform("d", 0, MAX_TPE_D, 1),
    hp.quniform("q", 0, MAX_TPE_Q, 1)
)

# Initialize trials object
trials = Trials()

# best = fmin(
#     fn = arima_tpe,
#     space = space, 
#     algo = tpe.suggest, 
#     max_evals = MAX_TPE_EVALS, 
#     trials=trials
# )

# print("Best: {}".format(best))

#------------------------------------------------------------------------------
# 4 - Prediction
#------------------------------------------------------------------------------

selectedP = 5
selectedD = 0
selectedQ = 1

model = ARIMA([x for x in diffs], order=(selectedP, selectedD, selectedQ))
model_fit = model.fit()
yhats = model_fit.forecast(step)

print(yhats)

p = prices[-1]
print(p)

for yhat in yhats:
    p += yhat
    print(p)
