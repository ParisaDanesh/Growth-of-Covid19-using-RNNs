import pandas
from pandas import read_csv, DataFrame, concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from tensorflow import keras
import numpy as np
import math
import csv

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forcast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        # print(df)
        # print('--'*50)
        # print(df.shift(1))
        # print('++'*50)
        # print(df.shift(-1))
        # print("**"*50)
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # print(agg)
        # exit()

        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        return agg

# load data
dataset = read_csv('raw_covid.csv', header=0, index_col=0)
output_ds = read_csv('raw_covid.csv', header=0, index_col=None)
values = dataset.values

# take date col for dashboard output
output = pandas.DataFrame(columns=None)
dates = output_ds.iloc[:,0]
dates = dates[366:]
output['Dates'] = dates

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
n_train_days = 365
train = values[:n_train_days, :]
test = values[n_train_days:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = keras.Sequential()
model.add(keras.layers.LSTM(70,
                            input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=30,
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

ys = inv_y.reshape(len(inv_y), 1)
output['Confirmed'] = ys
output.to_csv('output.csv', header=['Date', 'Confirmed'], index=False)

# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

