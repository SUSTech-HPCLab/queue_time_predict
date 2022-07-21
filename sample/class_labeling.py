import pickle

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

import numpy as np
from numpy import  *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sample.sample import Sample


class Labeler:
    def __init__(self, k, center=np.array([])):
        self.center = center
        self.k = k
        self.sampleList = list()
        self.scaler = None
        self.clusters= None

    # TODO

    def label_samples(self, sample_list):
        """
            给samples聚类，给Sample.class_label打标签
            :param sample_list: Sample数组
            :return: 打好标签的Sample数组
            """
        print('start label')
        random.shuffle(sample_list)

        processing_list = list()
        # 读取sample的各个特征,并将这些特征放入processingList中处理
        for sample in sample_list:
            array = list()
            array.append(sample.node)
            array.append(sample.request_time)
            processing_list.append(array)

        # preprocess = MinMaxScaler(feature_range=(0,1))
        # processing_list = preprocess.fit_transform(processing_list)
        # processingNp = np.array(processing_list)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(processing_list)
        processing_list = self.scaler.transform(processing_list)
        processing_np = np.array(processing_list)

        # preprocess = RobustScaler(quantile_range=(25.0,75.0))
        # preprocess.fit(processing_list)
        # processing_list = preprocess.transform(processing_list)
        # processing_np = np.array(processing_list)

        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(processing_np)
        self.center = kmeans.cluster_centers_
        y_kmeans = kmeans.predict(processing_np)

        num_list = []
        for i in range(0, self.k):
            num_list.append(0)

        for i in y_kmeans:
            num_list[i] = num_list[i] + 1

        for i in range(0, self.k):
            print('cluster', end=' ')
            print(i, ': ', num_list[i])

        for i in range(0, len(sample_list)):
            sample_list[i].class_label = y_kmeans[i]

        self.sampleList = (sample_list)

        # print(self.center)
        color_list = ['r', 'k', 'y', 'g', 'c', 'b', 'm', 'teal', 'dodgerblue',
                      'indigo', 'deeppink', 'pink', 'peru', 'brown', 'lime', 'darkorange']

        self.cluster = [[] for _ in range(self.k)]
        for sample in sample_list:
                array = list()
                array.append(sample.node)
                array.append(sample.request_time)
                array.append(sample.queue_job_sum)
                array.append(sample.run_job_sum)
                self.cluster[sample.class_label].append(array)

        for i in range(0,len(self.cluster)):
            self.cluster[i]=np.array(self.cluster[i])
            plt.scatter(self.cluster[i][:, 2], self.cluster[i][:, 3], cmap=plt.cm.Paired,color=color_list[i])


        # 画聚类中心
        # plt.scatter(self.center[:, 2], self.center[:, 3], marker='*', s=60)
        # for i in range(self.k):
        #     plt.annotate('中心' + str(i + 1), (self.center[i, 0], self.center[i, 1]))
        # plt.show()


        distortions = []
        for i in range(1, self.k + 1):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(processing_np)

            # inertia: Sum of squared distances of samples to their closest cluster center,
            # weighted by the sample weights if provided.
            distortions.append(km.inertia_)

        # plot
        plt.plot(range(1, self.k + 1), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        # plt.show()
        print('end label')
        return sample_list

    # TODO
    def label(self, sample):
        """
        给Sample.class_label打标签
        :param sample: Sample
        :return: int, 标签
        """

        array = list()
        array.append(sample.cpu_hours)
        array.append(sample.cpus)
        array.append(sample.queue_load)
        array.append(sample.system_load)
        vector = np.array(array)

        maxVector = self.scaler.data_max_
        minVector = self.scaler.data_min_

        X_std = (vector - minVector) / (maxVector - minVector)


        maxNumber = float("inf")
        for i in range(0, len(self.center)):
            temp = np.linalg.norm(self.center[i] - X_std)
            if temp <= maxNumber:
                sample.class_label = i
                maxNumber = temp
        self.sampleList.append(sample)
        return sample.class_label

    # TODO
    def load(self, file_path):
        """
        导入labeler
        :param file_path:
        """
        with open(file_path, 'rb') as text:
            tmp_list = pickle.load(text)
            self.center = pickle.load(text)

        for index, i in enumerate(tmp_list):
            tmp_sample = Sample()
            tmp_sample.id = index
            tmp_sample.cpu_hours = i[0]
            tmp_sample.cpus = i[1]
            tmp_sample.queue_load = i[2]
            tmp_sample.system_load = i[3]
            tmp_sample.actual_hour = i[4]
            tmp_sample.class_label = i[5]
            self.sampleList.append(tmp_sample)
        return self.sampleList

    # TODO
    def save(self, file_path):
        """
        导出labeler
        :param file_path:
        """
        save_list = []
        for i in self.sampleList:
            tmp_list = []
            tmp_list.append(i.cpu_hours)
            tmp_list.append(i.cpus)
            tmp_list.append(i.queue_load)
            tmp_list.append(i.system_load)
            tmp_list.append(i.actual_hour)
            tmp_list.append(i.class_label)
            save_list.append(tmp_list)
        with open(file_path, 'wb') as text:
            pickle.dump(save_list, text)
            pickle.dump(self.center, text)

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





