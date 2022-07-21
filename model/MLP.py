import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import init

from model.model import Model


class LinearNet(nn.Module):
    def __init__(self, n_feature, n_output):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(n_feature, 2)
        self.linear2 = nn.Linear(2, 2)
        self.output = nn.Linear(2, n_output)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        y = self.output(y)
        return y


class RegressionModel(Model):
    def __init__(self, sample_list, labeler, n_feature, n_output, raw_list):
        super().__init__(sample_list, labeler)
        self.n_feature = n_feature
        self.n_output = n_output
        self.model_list = []
        self.train_dataset = []
        self.test_dataset = []
        self.raw_list = raw_list


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
                if j.actual_sec <= 3600:
                    continue
                tem_train_list.append(float(j.node))
                tem_train_list.append(float(j.request_time))
                tem_train_list.append(float(j.cpu_sec))

                tem_train_list.append(float(j.queue_node_sum))
                tem_train_list.append(float(j.queue_request_time_sum))
                tem_train_list.append(float(j.queue_job_sum))
                tem_train_list.append(float(j.queue_cpu_sec_sum))
                tem_train_list.append(float(j.queue_time_sum))

                tem_train_list.append(float(j.run_node_sum))
                tem_train_list.append(float(j.run_request_time_sum))
                tem_train_list.append(float(j.run_job_sum))
                tem_train_list.append(float(j.run_cpu_sec_sum))
                tem_train_list.append(float(j.run_time_remain))

                train_list[j.class_label].append(tem_train_list)
                label_list[j.class_label].append(j.actual_sec)

        for i in range(0, self.labeler.k):

            n_epoch = 15
            batch_size = 32

            features = torch.from_numpy(np.array(train_list[i]))
            labels = torch.from_numpy(np.array(label_list[i]))
            labels = torch.tensor(labels, dtype=torch.double)

            dataset = data.TensorDataset(features, labels)
            data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

            net = LinearNet(self.n_feature, self.n_output).to(device='cuda')
            net = net.double()

            init.normal_(net.linear1.weight, mean=0, std=0.01)
            init.constant_(net.linear1.bias, val=0)
            init.normal_(net.linear2.weight, mean=0, std=0.01)
            init.constant_(net.linear2.bias, val=0)
            init.normal_(net.output.weight, mean=0, std=0.01)
            init.constant_(net.output.bias, val=0)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
            loss_fn = loss_fn.cuda()

            for epoch in range(1, n_epoch + 1):
                for x, y in data_iter:
                    x, y = x.cuda(), y.cuda()
                    output = net(x)
                    loss = loss_fn(output, y)
                    optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
                    loss.backward()
                    optimizer.step()
                print('epoch %d, loss: %f' % (epoch, loss.item()), end=' ------')
                print()
            net = net.to(device='cpu')
            self.model_list.append(net)
            print()

    # TODO
    def predict(self, sample):
        with torch.no_grad():
            pred = self.model_list[sample.class_label](
                torch.tensor(torch.Tensor([sample.node, sample.request_time, sample.cpu_sec, sample.queue_node_sum,
                                           sample.queue_request_time_sum, sample.queue_job_sum, sample.queue_cpu_sec_sum,
                                           sample.queue_time_sum, sample.run_node_sum, sample.run_request_time_sum,
                                           sample.run_job_sum, sample.run_cpu_sec_sum, sample.run_time_remain]), dtype=float))
            value = pred.tolist()[0]
            return value

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
        test = []
        sums = [0 for _ in range(6)]
        nums = [0 for _ in range(6)]

        for i in range(0, len(self.test_dataset)):
            for j in range(0, len(self.test_dataset[i])):
                test.append(self.test_dataset[i][j])
        rate = 0

        for i in test:
            if len(self.train_dataset[i.class_label]) <= 10:
                continue
            if i.actual_sec <= 3600:
                continue
            predict_time = max(0, self.predict(i))
            # predict_time = 2 ** predict_time - 1
            actual_time = i.actual_sec
            print(str(predict_time),str(actual_time))
            if actual_time <= 3600:  # 0-1
                sums[0] += abs(int(predict_time) - actual_time)
                nums[0] += 1

            elif actual_time <= 3600 * 3:  # 1-3
                sums[1] += abs(int(predict_time) - actual_time)
                nums[1] += 1
            elif actual_time <= 3600 * 6:  # 3-6
                sums[2] += abs(int(predict_time) - actual_time)
                nums[2] += 1
            elif actual_time <= 3600 * 12:  # 6-12
                sums[3] += abs(int(predict_time) - actual_time)
                nums[3] += 1
            elif actual_time <= 3600 * 24:  # 12-24
                sums[4] += abs(int(predict_time) - actual_time)
                nums[4] += 1
            else:
                sums[5] += abs(int(predict_time) - actual_time)
                nums[5] += 1

        avgs = [np.round(sums[i] / nums[i] / 3600, 2) for i in range(1, 6)]
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
