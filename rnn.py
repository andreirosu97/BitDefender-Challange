#%%
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout,Flatten
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

number_of_layers = 4
neurons_per_layer = 40

np.random.seed(555)

#%% load in data
df = pd.read_csv('challenge_train.csv', low_memory = False)
df = shuffle(df)
y = df['verdict']
X = df.drop(labels = ['md5','verdict'],axis = 'columns')
X.astype(np.float)
y.replace(to_replace={'trojan' : 1,'clean' : 0},inplace=True)
y.astype(np.float)

#%%
columns = []
values = []
elementI = 0
for label,content in X.items():
    columns.append(label)
    values.append('ft'+str(elementI))
    elementI += 1
features = dict(zip(columns,values))
X = X.rename(index = str, columns = features) 

#%% splitting data in train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

#%% Scaling
names = X.columns
scaler = preprocessing.StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=names)
x_valid = pd.DataFrame(x_valid, columns=names)
x_test = pd.DataFrame(x_test, columns=names)

#%% [markdown]
# # X TRAIN
x_train

#%% [markdown]
# # X TRAIN
x_valid

#%% [markdown]
# # X TRAIN
x_test

#%% [markdown]
# # PCA
pca = PCA(.95)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_valid = pca.transform(x_valid)
x_test = pca.transform(x_test)

names = [ 'pca'+str(i) for i in range(x_train.shape[1]) ]
x_train = pd.DataFrame(x_train, columns=names)
x_valid = pd.DataFrame(x_valid, columns=names)
x_test = pd.DataFrame(x_test, columns=names)

print(np.sum(pca.explained_variance_ratio_))

#%%
rf = RandomForestRegressor(n_estimators=128, n_jobs=-1, min_samples_leaf=3, max_depth=8, max_features=0.7, oob_score=True)
rf.fit(x_train, y_train)
print('Validation: ', r2_score(y_valid, rf.predict(x_valid)))
print('Training: ', r2_score(y_train, rf.predict(x_train)))
print('OOB', rf.oob_score_)
#%%
print(x_valid)
plt.scatter(y_valid, rf.predict(x_valid))

#%%
fimp = pd.DataFrame({'cols':x_train.columns, 'imp':rf.feature_importances_}).sort_values('imp', ascending=False)

#%%
fimp.plot('cols', 'imp', figsize=(16,9), legend=True)

#%%
fimp[:30].plot('cols', 'imp', 'barh', figsize=(16, 9), legend=False)

#%%constructing, training and evaluation of model
model = Sequential()
#%% input layer
model.add(Dense(units = x_train.shape[1],activation = 'relu', input_dim = x_train.shape[1]))

#%% hidden layers
for _ in range(number_of_layers):
	model.add(Dense(neurons_per_layer,activation = 'relu'))

#%% output layer
model.add(Dense(1,activation = 'sigmoid'))

#%% compilation, fitting and evaluating model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_data = (x_valid, y_valid))
(loss, accuracy) = model.evaluate(x_test, y_test, batch_size=64)
model.save_weights('network.h5')
print("Test accuracy: ", accuracy)
accuracy = history.history['acc']
epochs = range(len(accuracy))
plt.plot(epochs,accuracy,'ro',label='accuracy')
plt.grid()
plt.legend()
plt.show()

#%%