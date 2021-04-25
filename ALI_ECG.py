import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense,Flatten
from keras.layers import Conv1D
from keras import regularizers
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import os

def dp(row_num, all_data):
    plt.title(all_data.loc[row_num, "y"])
    sb.lineplot(y=all_data.iloc[row_num, 2:], x=[i_temp for i_temp in range(205)])
    plt.savefig("{}_{}.png".format(all_data.loc[row_num, "y"], row_num), dpi=300)
    plt.show()
def load_data():
    test_ = pd.read_csv(r"D:\ALI_ECG\testA.csv")
    train = pd.read_csv(r"D:\ALI_ECG\train.csv")

    train_x_temp = train["heartbeat_signals"].str.split(",", expand=True)
    train["y"] = train["label"].astype("int8")
    for i in train_x_temp.columns:
        train[i] = train_x_temp[i].astype("float64")
    train = train.drop(columns=["heartbeat_signals", "label"])

    train_data = train
    x = train_data.iloc[:, 2:]
    y = train["y"]

    test_temp = test_["heartbeat_signals"].str.split(",", expand=True)
    test = pd.DataFrame()
    for i in test_temp.columns:
        test[i] = test_temp[i].astype("float64")
    test = np.expand_dims(test,axis=2)
    return x,y,test,test_
    # smo = SMOTE(random_state=42)
    # train_data, y = smo.fit_sample(train_data, y)
    # print(collections.Counter(y))


    # x = train_data.iloc[:, 2:]
    # y = (OneHotEncoder().fit_transform(y.values.reshape(-1, 1))).toarray()
    # # 直奔主题，全数据集训练
    # x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8)
    # x_train = np.expand_dims(x_train,axis=2)
    # x_val = np.expand_dims(x_val,axis=2)
    # # y_train = np.expand_dims(y_train,axis=2)
    # # y_val = np.expand_dims(y_val,axis=2)
    # print(x_train.shape)
    # print(x_val.shape)
    # print(y_train)
    #
    # test_temp = test_["heartbeat_signals"].str.split(",", expand=True)
    # test = pd.DataFrame()
    # for i in test_temp.columns:
    #     test[i] = test_temp[i].astype("float64")
    # test = np.expand_dims(test,axis=2)
    # return x_train,x_val,y_train,y_val,test,test_


def build_network():
    model = Sequential()
    model.add(Conv1D(5, 64, kernel_regularizer=regularizers.l2(0.2), input_shape=(205, 1)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(10, 32, kernel_regularizer=regularizers.l2(0.2)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(20, 16, kernel_regularizer=regularizers.l2(0.2)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2))

    # 压平层
    model.add(Flatten())
    # 来一个全连接层
    model.add(Dense(30))
    model.add(LeakyReLU(alpha=0.05))

    # 再来一个全连接层
    model.add(Dense(20))
    model.add(LeakyReLU(alpha=0.05))

    # 最后为分类层
    model.add(Dense(4, activation='softmax'))

    from keras.optimizers import SGD
    optimizer = SGD(lr=0.003, momentum=0.7, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    return model

def lr_schedule(epoch, lr):
    if epoch > 70 and \
            (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr

def plot(history):
    """Plot performance curve"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], "r-", history["val_loss"], "b-", linewidth=0.5)
    axes[0].set_title("Loss")
    axes[1].plot(history["accuracy"], "r-", history["val_accuracy"], "b-", linewidth=0.5)
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.show()



def abs_sum(y_true,y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    loss = sum(sum(abs(y_pred - y_true)))
    return loss

def mutil_loss(y_true,y_pred):

    loss1 = abs(y_pred - y_true)
    loss2 = categorical_crossentropy(y_true, y_pred)
    return loss1+loss2

cv_score = []
if __name__ == "__main__":
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    label0 = []
    label1 = []
    label2 = []
    label3 = []
    data_x,labels, test,test_ = load_data()

    for i in range(len(labels)):
        if labels[i] == 0:
            data0.append(data_x.iloc[i])
            label0.append(0)
        elif labels[i] == 1:
            data1.append(data_x.iloc[i])
            label1.append(1)
        elif labels[i] == 2:
            data2.append(data_x.iloc[i])
            label2.append(2)
        elif labels[i] == 3:
            data3.append(data_x.iloc[i])
            label3.append(3)

    max_len = len(label0)
    count = 0
    for i in range(0, max_len, 12866):
        count +=  1
        e = i + 12866
        if e > max_len:
            e = max_len
        Train_x = data0[i:e] + data1[:] + data2[:] + data3[:]
        Train_y = label0[i:e] + label1[:] + label2[:] + label3[:]
        Train_x = np.array(Train_x)
        Train_y = np.array(Train_y)
        # print(Train_x.shape)
        # print(Train_y.shape)
        shuffle_ix = np.random.permutation(np.arange(len(Train_x)))
        Train_x = Train_x[shuffle_ix]
        Train_y = Train_y[shuffle_ix]
        # Train_x = np.expand_dims(Train_x, axis=2)

        print(Train_x.shape)
        print(Train_y.shape)
        x_train, x_val, y_train, y_val = train_test_split(Train_x, Train_y, train_size=0.8)

        smo = SMOTE(random_state=42)
        x_train, y_train = smo.fit_sample(x_train, y_train)

        x_train = np.expand_dims(x_train, axis=2)
        x_val = np.expand_dims(x_val, axis=2)
        y_train = to_categorical(y_train, num_classes=4)  # Convert to one-hot
        y_val = to_categorical(y_val, num_classes=4)



        model = build_network()
        model.summary()
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        lr_scheduler = LearningRateScheduler(lr_schedule)
        model_name = 'model_cnn9.h5'
        checkpoint = ModelCheckpoint(filepath=model_name,
                                     monitor='val_categorical_accuracy', mode='max',
                                     save_best_only='True')

        callback_lists = [lr_scheduler, checkpoint]
        history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val),
                            callbacks=callback_lists)
        model.save(os.path.join("models", "smote_model_ecg.h5"))
        loss, accuracy = model.evaluate(x_val, y_val)  # test the model
        print("Test loss: ", loss)
        print("Accuracy: ", accuracy)
        plot(history.history)
        y_val_pre = model.predict(x_val)
        score = abs_sum(y_val,y_val_pre)
        print('score:',score)
        cv_score.append(score)

        y_pre = model.predict(test)

        pre_file = pd.DataFrame()
        pre_file["id"] = test_["id"]
        pre_file["label_0"] = y_pre[:, 0]
        pre_file["label_1"] = y_pre[:, 1]
        pre_file["label_2"] = y_pre[:, 2]
        pre_file["label_3"] = y_pre[:, 3]
        csv_name1 = "0420_submit_cross_5k_smote1_" + str(count) + ".csv"
        pre_file.to_csv(csv_name1, index=None)
        # 优化结果数据
        out_file = pd.DataFrame()
        a = (OneHotEncoder().fit_transform(y_pre.argmax(axis=1).reshape(-1, 1))).toarray()
        print(a.shape)
        out_file["id"] = test_["id"]
        out_file["label_0"] = a[:, 0]
        out_file["label_1"] = a[:, 1]
        out_file["label_2"] = a[:, 2]
        out_file["label_3"] = a[:, 3]
        # 导出
        csv_name2 = "0420_submit_cross_5k_smote2_"+str(count)+".csv"
        out_file.to_csv(csv_name2, index=None)


