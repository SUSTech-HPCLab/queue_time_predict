import numpy as np
import math
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from model.model import Model


class SklearnLogisticRegressionModel(Model):
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
                tem_train_list.append(math.log2(j.cpu_hours + 1))
                tem_train_list.append(j.cpus)
                tem_train_list.append(j.queue_load)
                tem_train_list.append(math.log2(j.system_load + 1))

                tem_train_list.append(j.future_load)
                tem_train_list.append(j.future_node_load)
                tem_train_list.append(j.future_requested_sec_load)

                train_list[j.class_label].append(tem_train_list)
                label_list[j.class_label].append(self.classification(j.actual_sec))

        for i in range(0, self.labeler.k):
            if len(train_list[i]) <= 10:
                self.model_list.append(None)
                continue
            X_train, X_test, y_train, y_test = train_test_split(
                train_list[i], label_list[i],
                train_size=0.8, test_size=0.2, random_state=188
            )

            log_model = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=10000)

            # 使用训练数据来学习（拟合），不需要返回值，训练的结果都在对象内部变量中
            log_model.fit(X_train, y_train)
            self.model_list.append(log_model)
            pred_test = log_model.predict(X_test)
            acu = accuracy_score(y_test, pred_test)  # 准确率
            # print('准确率', end=': ')
            # print(acu*100,end='%')
            # print()

    # TODO
    def predict(self, sample):
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
        result = selected_model.predict(return_list)
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

            index = count[self.sample_list[i].class_label] / 10 * 10

            if len(self.train_dataset[self.sample_list[i].class_label]) <= index:
                self.train_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

            else:
                self.test_dataset[self.sample_list[i].class_label].append(self.sample_list[i])

    def test(self):
        test = []

        for i in range(0, len(self.test_dataset)):
            for j in range(0, len(self.test_dataset[i])):
                test.append(self.test_dataset[i][j])
        all_num = 0
        all_true = 0
        all_near = 0
        num = [0] * 6
        true = [0] * 6
        near = [0] * 6
        for i in test:
            if len(self.train_dataset[i.class_label]) <= 10:
                continue
            predict_class = self.predict(i)
            actual_class = self.classification(i.actual_sec)
            if abs(predict_class - actual_class) <= 1:
                near[actual_class] = near[actual_class] + 1
                all_near = all_near + 1
            all_num = all_num + 1
            num[actual_class] = num[actual_class] + 1
            if predict_class == actual_class:
                all_true = all_true + 1
                true[actual_class] = true[actual_class] + 1
        print('准确率', end=': ')
        print((all_true / all_num) * 100, end='%')
        print()

        print('相邻', end=': ')
        print((all_near / all_num) * 100, end='%')
        print()
        for i in range(0, len(num)):
            print(i, true[i] / num[i], end=' ')
            print('相邻', end=': ')
            print(near[i] / num[i])

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

    def classification(self, sec):
        if sec <= 3600:  # 0-1
            return 0
        elif sec <= 3600 * 3:  # 1-3
            return 1
        elif sec <= 3600 * 6:  # 3-6
            return 2
        elif sec <= 3600 * 12:  # 6-12
            return 3
        elif sec <= 3600 * 24:  # 12-24
            return 4
        else:
            return 5