import numpy as n
import matplotlib.pyplot as plt
import numpy as np
import pandas as p
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


data = p.read_csv('BTC-USD.csv', date_parser=True)

# Letting data up to be 31-Oct-2021 be the training set
training_data = data[data['Date'] <= '2021-10-31'].copy()
# Test data will be the data from 1-Nov-2021 to 21-Nov-2021
test_data = data[data['Date'] > '2021-10-31'].copy()
# Formatting the test and training tables by dropping irrelevant columns

training_data = training_data.drop(['Date', 'Adj Close'], axis=1)

# Now we normalize the training data using MinMaxScaler

scaler = MinMaxScaler()
train_data = scaler.fit_transform(training_data)
X_train = []
Y_train = []

for i in range(60, train_data.shape[0]):
    X_train.append(train_data[i - 60:i])
    Y_train.append(train_data[i, 0])

X_train = n.array(X_train)
Y_train = n.array(Y_train)

# RNN intialization
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=3, batch_size=25, validation_split=0.1)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

part_60_days = training_data.tail(60)
df = part_60_days.append(test_data, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
df.head()
inputs = scaler.transform(df)

X_test = []
Y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    Y_test.append(inputs[i, 0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

regressor = LinearRegression()
Y_pred = regressor.predict(X_test)

scale = 1/scaler.scale_[0]
Y_test = Y_test*scale
Y_pred = Y_pred*scale

plt.figure(figsize=(14, 5))
plt.plot(Y_test, color='red', label='Real Bitcoin Price')
plt.plot(Y_pred, color='green', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
