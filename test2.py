# Create your first MLP in Keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import model_from_json
import h5py
import copy
import random

numpy.random.seed(7)
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X_train_10d = dataset[:, 0:8]
Y_train_10d = dataset[:, 8]
size_10d=X_train_10d.shape[0]
X_train_1d = dataset[345:385, 0:8]
Y_train_1d = dataset[345:385, 8]
size_1d=X_train_1d.shape[0]
X_train_9d = dataset[:345, 0:8]
Y_train_9d = dataset[:345, 8]
size_9d=X_train_9d.shape[0]
#X_test = dataset[384:, 0:8]
#Y_test = dataset[384:, 8]
X_test = dataset[:, 0:8]
Y_test = dataset[:, 8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.save_weights("result/init.h5")

# 10d train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_json = model.to_json()
with open("result/model_10d.json", "w") as json_file:
    json_file.write(model_json)
model.load_weights("result/init.h5")
model.fit(X_train_10d, Y_train_10d, epochs=150, batch_size=10)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save_weights("result/10d.h5")

# 1d train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights("result/init.h5")
model.fit(X_train_1d, Y_train_1d, epochs=150, batch_size=10)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save_weights("result/1d.h5")

# 9d train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights("result/init.h5")
model.fit(X_train_9d, Y_train_9d, epochs=150, batch_size=10)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save_weights("result/9d.h5")

#load 1d
json_file = open('result/model_10d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loaded_model.load_weights("result/1d.h5")
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("1d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#load 10d
loaded_model.load_weights("result/10d.h5")
weights_10d=loaded_model.get_weights()
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("10d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#load 9d
loaded_model.load_weights("result/9d.h5")
weights_9d=loaded_model.get_weights()
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("9d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#load no learn
loaded_model.load_weights("result/init.h5")
weights_1d=loaded_model.get_weights()
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("no learn result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

nine_count = 0.
intdiv_1d_9d_weights=loaded_model.get_weights()
print(intdiv_1d_9d_weights[0][1][5])
for i in range(len(weights_10d)):
    for j in range(len(weights_10d[i])):
        print(len(weights_10d[i][j].shape))
        for k in range(len(weights_10d[i][j])):
            myrd = random.uniform(0, 1)
            if myrd <= 0.9:
                intdiv_1d_9d_weights[i][j][k] = numpy.copy(weights_9d[i][j][k])
                nine_count = nine_count + 1
            else:
                intdiv_1d_9d_weights[i] = numpy.copy(weights_1d[i][j][k])
    # intdiv_1d_9d_weights[i]=numpy.copy((weights_9d[i]*size_9d+weights_1d[i]*size_1d)/(size_9d+size_1d))
loaded_model.set_weights(intdiv_1d_9d_weights)
loaded_model.save_weights("result/intdiv_1d_9d.h5")


#load intdiv_9d_1d
loaded_model.load_weights("result/intdiv_1d_9d.h5")
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("internally divided 9 to 1 result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print(nine_count/len(weights_10d))
print(len(weights_10d))
print(len(weights_1d))
print(len(weights_9d))
print(len(intdiv_1d_9d_weights))