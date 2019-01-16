# Create your first MLP in Keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import model_from_json
import h5py
import copy
import random
from math import exp

def sigmoid(x):
  return 1 / (1 + exp(-x))

def inverse_sigmoid(x):
    return numpy.log(x/(1-x))

divisionRatio = 0.1
testSetRatio = 0.1

_noLearn_score_lst = []
_1d_score_lst = []
_9d_score_lst = []
_10d_score_lst = []
_linear_div_score_lst = []
_stoch_div_score_lst = []

for tries in range(5):
    # numpy.random.seed(7)
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    numpy.random.shuffle(dataset)
    print(dataset.shape)
    size_10d = int(dataset.shape[0] * (1-testSetRatio))
    size_9d = int(dataset.shape[0] * (1-testSetRatio) * (1-divisionRatio))
    print('size_10d:', size_10d)
    print('size_9d:', size_9d)

    X_train_10d = dataset[:size_10d, 0:8]
    Y_train_10d = dataset[:size_10d, 8]
    X_train_1d = dataset[size_9d:size_10d, 0:8]
    Y_train_1d = dataset[size_9d:size_10d, 8]
    size_1d=X_train_1d.shape[0]
    print('size_1d:' ,size_1d)
    X_train_9d = dataset[:size_9d, 0:8]
    Y_train_9d = dataset[:size_9d, 8]

    X_test = dataset[size_10d:dataset.shape[0], 0:8]
    Y_test = dataset[size_10d:dataset.shape[0], 8]
    print(X_test.shape)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model.save_weights("result/init.h5")

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_json = model.to_json()
    with open("result/model_10d.json", "w") as json_file:
        json_file.write(model_json)

    # 1d train
    model.load_weights("result/init.h5")
    model.fit(X_train_1d, Y_train_1d, epochs=120, batch_size=10, verbose=0)
    # score = model.evaluate(X_test, Y_test)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    model.save_weights("result/1d.h5")

    #10d train
    model.load_weights("result/init.h5")
    model.fit(X_train_10d, Y_train_10d, epochs=120, batch_size=10, verbose=0)
    # score = model.evaluate(X_test, Y_test)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    model.save_weights("result/10d.h5")

    # 9d train
    model.load_weights("result/init.h5")
    model.fit(X_train_9d, Y_train_9d, epochs=120, batch_size=10, verbose=0)
    score = model.evaluate(X_test, Y_test)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    model.save_weights("result/9d.h5")

    #load model
    json_file = open('result/model_10d.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #load and eval 1d
    loaded_model.load_weights("result/1d.h5")
    weights_1d=loaded_model.get_weights()
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("1d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _1d_score_lst.append(score[1]*100)

    #load and eval 10d
    loaded_model.load_weights("result/10d.h5")
    weights_10d=loaded_model.get_weights()
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("10d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _10d_score_lst.append(score[1]*100)

    #load and eval 9d
    loaded_model.load_weights("result/9d.h5")
    weights_9d=loaded_model.get_weights()
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("9d result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _9d_score_lst.append(score[1]*100)

    #load no learn
    loaded_model.load_weights("result/init.h5")
    # print(loaded_model.get_weights()[0][0])
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("no learn result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _noLearn_score_lst.append(score[1]*100)

    # loaded_model.load_weights("result/9d.h5")
    # print(loaded_model.get_weights()[0][0])
    # loaded_model.load_weights("result/init.h5")
    # print(loaded_model.get_weights()[0][0])

    linear_intdiv_1d_9d_weights=loaded_model.get_weights()
    stoch_intdiv_1d_9d_weights=loaded_model.get_weights()

    for i in range(len(weights_10d)):
        assert len(weights_10d[i].shape) == 1 or len(weights_10d[i].shape) == 2
        if len(weights_10d[i].shape) == 1:
            for j in range(len(weights_10d[i])):
                linear_intdiv_1d_9d_weights[i][j] = (weights_9d[i][j] * size_9d + weights_1d[i][j] * size_1d) / (size_9d+size_1d)

                myrd = random.uniform(0, 1)
                if myrd <= 0.9:
                    stoch_intdiv_1d_9d_weights[i][j] = weights_9d[i][j]
                else:
                    stoch_intdiv_1d_9d_weights[i][j] = weights_1d[i][j]
        else:
            for j in range(len(weights_10d[i])):
                for k in range(len(weights_10d[i][j])):
                    linear_intdiv_1d_9d_weights[i][j][k] = (weights_9d[i][j][k] * size_9d + weights_1d[i][j][k] * size_1d) / (size_9d + size_1d)

                    myrd = random.uniform(0, 1)
                    if myrd <= 0.9:
                        stoch_intdiv_1d_9d_weights[i][j][k] = weights_9d[i][j][k]
                    else:
                        stoch_intdiv_1d_9d_weights[i][j][k] = weights_1d[i][j][k]
        # print(intdiv_1d_9d_weights[i])
        # print(intdiv_1d_9d_weights[i].shape)
        # print(len(intdiv_1d_9d_weights[i].shape))
        # # print(intdiv_1d_9d_weights[i].shape[0])
        # # print(intdiv_1d_9d_weights[i].shape[1])
        # print('\n')
        # for j in range(len(weights_10d[i])):
        #     print(len(weights_10d[i][j].shape))
            # for k in range(len(weights_10d[i][j])):
                # myrd = random.uniform(0, 1)
                # if myrd <= 0.9:
                #     intdiv_1d_9d_weights[i][j][k] = numpy.copy(weights_9d[i][j][k])
                #     nine_count = nine_count + 1
                # else:
                #     intdiv_1d_9d_weights[i][j][k] = numpy.copy(weights_1d[i][j][k])
        # intdiv_1d_9d_weights[i]=numpy.copy((weights_9d[i]*size_9d+weights_1d[i]*size_1d)/(size_9d+size_1d))

    # test weights following above policy

    # linear
    loaded_model.set_weights(linear_intdiv_1d_9d_weights)
    loaded_model.save_weights("result/linear_intdiv_1d_9d.h5")
    loaded_model.load_weights("result/linear_intdiv_1d_9d.h5")
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("linear_internally divided result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _linear_div_score_lst.append(score[1]*100)

    # semi stochastic
    loaded_model.set_weights(stoch_intdiv_1d_9d_weights)
    loaded_model.save_weights("result/semiStoch_intdiv_1d_9d.h5")
    loaded_model.load_weights("result/semiStoch_intdiv_1d_9d.h5")
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print("stoch_internally divided result: %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    _stoch_div_score_lst.append(score[1]*100)

    print('----------------------------------------------------------------------------------------------------------------------------')

    # for i in range(2):
    #     for j in range(5):
    #         print('---------------------------------------------------------------')
    #         print(weights_9d[0][i][j])
    #         print(weights_1d[0][i][j])
    #         print(linear_intdiv_1d_9d_weights[0][i][j])
    #         print(semiStoch_intdiv_1d_9d_weights[0][i][j])
def get_avg(lst):
    sum = 0.
    for number in lst:
        sum = sum + number
    return sum/len(lst)

def get_derv(lst):
    assert len(lst)>=2
    avg = get_avg(lst)
    sum = 0.
    for number in lst:
        sum = sum + (number - avg)**2
    return (sum / (len(lst) - 1))**0.5

print('_noLearn_score_lst:', _noLearn_score_lst)
print('_1d_score_lst:', _1d_score_lst)
print('_9d_score_lst:', _9d_score_lst)
print('_10d_score_lst:', _10d_score_lst)
print('_linear_div_score_lst:', _linear_div_score_lst)
print('_stoch_div_score_lst', _stoch_div_score_lst)

print('noLearn', get_avg(_noLearn_score_lst), 'noLearn derv', get_derv(_noLearn_score_lst))
print('1d avg', get_avg(_1d_score_lst), '1d derv', get_derv(_1d_score_lst))
print('9d avg', get_avg(_9d_score_lst), '9d derv', get_derv(_9d_score_lst))
print('10d avg', get_avg(_10d_score_lst), '10d derv', get_derv(_10d_score_lst))
print('linear avg', get_avg(_linear_div_score_lst), 'linear derv', get_derv(_linear_div_score_lst))
print('stochastic avg', get_avg(_stoch_div_score_lst), 'stochastic derv', get_derv(_stoch_div_score_lst))




