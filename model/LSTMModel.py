import numpy as np
import math

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from model.model import Model
from sklearn import tree
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM




class LSTMModel(Model):
    def __init__(self, sample_list, labeler, n_feature, n_output, raw_list):
        super().__init__(sample_list, labeler)
        self.n_feature = n_feature
        self.n_output = n_output
        self.model_list = []
        self.train_dataset = []
        self.test_dataset = []
        self.raw_list = raw_list
        self.timestep=100

    """
       我们以数据集 训练，测试比例 8:2的比例进行来来行训练与测试，
       同时将训练好的模型按照聚类的类别保存到对应下标的model_list中
       """

    def train(self):
        train_list = []
        label_list = []
        for i in range(0, self.labeler.k):
            train = list()
            label = list()
            train_list.append(train)
            label_list.append(label)

        for i in range(0, self.labeler.k):
            for j in self.train_dataset[i]:
                tem_train_list = list()
                tem_train_list.append(math.log2(j.cpu_hours + 1))
                tem_train_list.append(j.cpus)
                tem_train_list.append(j.queue_load)
                tem_train_list.append(math.log2(j.system_load + 1))

                tem_train_list.append(j.future_load)
                tem_train_list.append(j.future_node_load)
                tem_train_list.append(j.future_requested_sec_load)

                train_list[j.class_label].append(tem_train_list)
                label_list[j.class_label].append(math.log2(j.actual_sec+1))
                # label_list[j.class_label].append((j.actual_sec ))

        label_list = np.array(label_list)
        data = np.array([[label_list[0][i]] for i in range(len(label_list[0]))])

        sampleList=[]

        for i in range(0,len(label_list[0])) :
            if i !=0:
                sampleList.append([label_list[0][i]])

        sampleList.append([0])
        sample= np.array([[sampleList[i]] for i in range(len(sampleList))])
        # sample= sample.reshape((sample.shape[0], 1, 1))
        # data = data.reshape((data.shape[0], 1, data.shape[1]))

        generator = TimeseriesGenerator(data, data, length=self.timestep, batch_size=1)
        inputdata=[]
        inputsample=[]
        for i in range(len(generator)):
            x, y = generator[i]
            inputdata.append(x)
            inputsample.append(y)
            # print('%s => %s' % (x, y))
        inputdata=np.array(inputdata)
        inputsample=np.array(inputsample)
        inputdata= inputdata.reshape((inputdata.shape[0], self.timestep, inputdata.shape[1]))
        for i in range(0, self.labeler.k):

            lstm_model = Sequential()
            # lstmInstance = LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2]))
            lstmInstance = LSTM(10, input_shape=(self.timestep, data.shape[1]))
            lstm_model.add(lstmInstance)
            lstm_model.add(Dense(1))
            # 2. 编译网络
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            # 3. 训练网络
            # lstm_model.fit(generator, steps_per_epoch=1,epochs=10, verbose=0)
            history = lstm_model.fit(inputdata, inputsample, batch_size=72, epochs=10, verbose=0)
            # print(lstm_model.predict(np.array([[1,2,3]])))
            # print(lstm_model.predict(np.array([[10,11,12]])))
            # print(lstm_model.predict(np.array([[100,110,120]])))
            # print(lstm_model.predict(np.array([[10000,11000,12000]])))
            # print(lstm_model.predict(np.array([[1000000,1200000,1400000]])))

            self.model_list.append(lstm_model)



    # TODO
    def predict(self, sample,input):
        return_list = list()
        tmp_list = list()

        tmp_list.append(math.log2(sample.cpu_hours + 1))
        tmp_list.append(sample.cpus)
        tmp_list.append(sample.queue_load)
        tmp_list.append(math.log2(sample.system_load + 1))

        tmp_list.append(sample.future_load)
        tmp_list.append(sample.future_node_load)
        tmp_list.append(sample.future_requested_sec_load)

        tmp_np = np.array(tmp_list)
        return_list.append(tmp_np)
        return_list = np.array(return_list)
        selected_model = self.model_list[sample.class_label]
        # return_list = return_list.reshape((return_list.shape[0],  1, return_list.shape[1]))
        result = selected_model.predict(input)

        return result[0]

    # TODO
    def save(self, file_path):
        pass

    # TODO
    def load(self, file_path):
        pass

    def create_dataset(self):
        """
        对于输入的sample_list进行训练集与测试集的划分。
        我们是采取训练训练集：测试集 = 8：2的比例进行划分。
        注意，这里进入的sample_list已经进行过打乱。
        """
        for i in range(0, self.labeler.k):
            train = list()
            test = list()
            self.train_dataset.append(train)
            self.test_dataset.append(test)
        count = [0] * self.labeler.k
        for i in range(0, len(self.sample_list)):
            count[self.sample_list[i].class_label] = count[self.sample_list[i].class_label] + 1

        for i in range(0, len(self.sample_list)):

            index = count[self.sample_list[i].class_label] / 10 * 8

            if len(self.train_dataset[self.sample_list[i].class_label]) <= index:
                self.train_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

            else:
                self.test_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

    def test(self):
        print("start lstm test")
        test = []
        sums = [0 for _ in range(6)]
        nums = [0 for _ in range(6)]

        for i in range(0, len(self.test_dataset)):
            for j in range(0, len(self.test_dataset[i])):
                test.append(self.test_dataset[i][j])
        rate = 0
        count=0
        for i in test:
            # print(count)

            if len(self.train_dataset[i.class_label]) <= 10:
                count = count + 1
                continue
            if count < self.timestep:
                count = count + 1
                continue

            input= np.array([[math.log2(test[count-self.timestep+i].actual_sec+1)] for i in range(0,self.timestep)])
            input = input.reshape((1, self.timestep, input.shape[1]))
            # input = np.array(
            #     [[test[count - self.timestep + i].actual_sec ] for i in range(0, self.timestep)])
            # input=[[10+count*1000]]
            a=self.model_list[0].predict(input)
            predict_time = max(0, self.model_list[0].predict(input))
            predict_time = 2 ** predict_time - 1
            actual_time = i.actual_sec
            print(str(predict_time),str(actual_time))
            execution_time = self.raw_list[i.id].end_ts - self.raw_list[i.id].start_ts
            rate = abs(predict_time - actual_time) / (actual_time + execution_time) + rate
            count = count + 1
            if actual_time <= 3600:  # 0-1
                sums[0] += abs(predict_time - actual_time)
                nums[0] += 1

            elif actual_time <= 3600 * 3:  # 1-3
                sums[1] += abs(predict_time - actual_time)
                nums[1] += 1
            elif actual_time <= 3600 * 6:  # 3-6
                sums[2] += abs(predict_time - actual_time)
                nums[2] += 1
            elif actual_time <= 3600 * 12:  # 6-12
                sums[3] += abs(predict_time - actual_time)
                nums[3] += 1
            elif actual_time <= 3600 * 24:  # 12-24
                sums[4] += abs(predict_time - actual_time)
                nums[4] += 1
            else:
                sums[5] += abs(predict_time - actual_time)
                nums[5] += 1


        avgs = [np.round(sums[i] / nums[i] / 3600, 2) for i in range(6)]
        print(avgs)
        AAE = np.round(sum(sums) / sum(nums) / 3600, 2)
        print('AAE :', end=' ')
        print(AAE)
        PPE = rate / sum(nums)
        print('PPE :', end=' ')
        print(PPE)
        return AAE, PPE

    def label_queue_name(self):
        queue_name_list = []

        for i in self.sample_list:
            if i.queue_name not in queue_name_list:
                queue_name_list.append(i.queue_name)

        for i in self.sample_list:
            i.class_label = queue_name_list.index(i.queue_name)

        self.labeler.k = len(queue_name_list)

    def without_label(self):
        self.labeler.k = 1
        for i in self.sample_list:
            i.class_label = 1
