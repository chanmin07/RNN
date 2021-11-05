# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set - 5년간의 구글의 주식가격 (2012~2016), 2017년 1월의 주식가격 예측 
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # 1-th column data(oprn column)만 읽어서 training_set에 넣어라. numpy.array: 1258 x 1 

# Feature Scaling - Standardization ( X - mu ) / std or Normalization (X - min_X) / (max_X - min_X): Normalization here
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))               # sc: object in MinMaxScaler class
training_set_scaled = sc.fit_transform(training_set)    

# Number of Timesteps: Creating a data structure with 60 timesteps (check previous 60- timesteps) and 1 output
# Too small - Overfitting - learn nothing
X_train = []        # stock prices at 60 previous steps 
y_train = []        # stock price at the next step
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape( X_train, (X_train.shape[0], X_train.shape[1], 1)) # The third dim - 다른 요소들을 검사하고자 하는 경우에 유용하다. 
    # Check keras documentation for dim - Layer API - Recurrent Layer - RNN Base
    # Input shape - 
    # N-D tensor with shape [batch_size, timesteps, ...] (1198, 60, 1) or [timesteps, batch_size, ...] when time_major is True.
    
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential    # To create NN Object representing a seq. of layers
from keras.layers import Dense         # To add output layer
from keras.layers import LSTM          # To add LSTM layer
from keras.layers import Dropout       # For Regularization

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 
        # number of LSTM cells, True = if we want to add more LSTM layer 
regressor.add(Dropout(0.2))     # To ignore 20% neurons in LSTM layer in each iteration (forward/backpro) 

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))    # 이 두번째 layer는 이미 입력데이터의 크기를 알고 있다.  
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))     # This is the last LSTM layer
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Check Keras Document

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)      # 3000초, epochs = 반복횟수, batch_size = 32 Total number of training examples present in a single batch
    # To avoid overfitting, do not seek for too small loss
    
# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017.01
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017 - scale input !!!
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)   
    # trick
    #inputs.shape
    #Out[31]: (80,)

    #inputs = inputs.reshape(-1,1)

    #inputs.shape
    #Out[33]: (80, 1)
inputs = sc.transform(inputs)
    # 기존의 스케일 방법으로 스케일 하고 싶기 ㄸ째문 - 1보다 큰 숫자도 있다. 
    
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Back-scaling 

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
