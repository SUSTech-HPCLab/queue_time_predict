import numpy as np
import math
# import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from model.model import Model
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
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
                tem_train_list.append(j.node)
                tem_train_list.append(j.request_time)
                tem_train_list.append(j.cpu_sec)

                tem_train_list.append(j.queue_node_sum)
                tem_train_list.append(j.queue_request_time_sum)
                tem_train_list.append(j.queue_job_sum)
                tem_train_list.append(j.queue_cpu_sec_sum)
                tem_train_list.append(j.queue_time_sum)

                tem_train_list.append(j.run_node_sum)
                tem_train_list.append(j.run_request_time_sum)
                tem_train_list.append(j.run_job_sum)
                tem_train_list.append(j.run_cpu_sec_sum)
                tem_train_list.append(j.run_time_remain)

                train_list[j.class_label].append(tem_train_list)
                label_list[j.class_label].append(self.classification(j.actual_sec))

        for i in range(0, self.labeler.k):
            if len(train_list[i]) <= 10:
                self.model_list.append(None)
                continue
            X_train, X_test, y_train, y_test = train_test_split(
                train_list[i], label_list[i],
                train_size=0.7, test_size=0.3, random_state=0
            )

            forest = RandomForestClassifier(n_estimators=4000, random_state=0, n_jobs=-1)
            forest.fit(X_train, y_train)

            importances = forest.feature_importances_
            feature = ["node", "request_time", "cpu_sec", "queue_node_sum", "queue_request_time_sum", "queue_job_sum", "queue_cpu_sec_sum",
                       "queue_time_sum", "run_node_sum", "run_request_time_sum", "run_job_sum", "run_cpu_sec_sum", "run_time_remain"]
            print("mira results")
            for i in range(0, len(importances)):
                print(feature[i],importances[i])
            # for f in range(X_train.shape[1]):
                # print("%2d) %-*s %f" % (f + 1, 30, importances[indices[f]]))

    # TODO
    def predict(self, sample):
        return_list = list()
        tmp_list = list()

        tmp_list.append(sample.node)
        tmp_list.append(sample.request_time)
        tmp_list.append(sample.cpu_sec)

        tmp_list.append(sample.queue_node_sum)
        tmp_list.append(sample.queue_request_time_sum)
        tmp_list.append(sample.queue_job_sum)
        tmp_list.append(sample.queue_cpu_sec_sum)
        tmp_list.append(sample.queue_time_sum)

        tmp_list.append(sample.run_node_sum)
        tmp_list.append(sample.run_request_time_sum)
        tmp_list.append(sample.run_job_sum)
        tmp_list.append(sample.run_cpu_sec_sum)
        tmp_list.append(sample.run_time_remain)

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
        sums = [0 for _ in range(6)]
        nums = [0 for _ in range(6)]

        for i in range(0, len(self.test_dataset)):
            for j in range(0, len(self.test_dataset[i])):
                test.append(self.test_dataset[i][j])
        rate = 0

        for i in test:
            if len(self.train_dataset[i.class_label]) <= 10:
                continue
            predict_time = max(0, self.predict(i))
            predict_time = 2 ** predict_time - 1
            actual_time = i.actual_sec
            # print(str(predict_time),str(actual_time))
            # execution_time = self.raw_list[i.id].end_ts - self.raw_list[i.id].start_ts
            # rate = abs(predict_time - actual_time) / (actual_time + execution_time) + rate
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
