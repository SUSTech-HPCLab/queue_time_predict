class Model:
    def __init__(self, sample_list, labeler=None):
        self.sample_list = sample_list
        self.labeler = labeler
    def train(self):
        """
        主函数，在这里运行
        """
        pass

    def predict(self, sample):
        """
        预测一个sample的actual_hour
        :param sample: Sample
        :return: double, 输出的时间
        """
        pass

    def save(self, file_path):
        """
        保存model
        :param file_path:
        :return:
        """
        pass

    def load(self, file_path):
        """
        恢复model。不需要恢复sampler_list和labeler
        :param file_path:
        :return:
        """
        pass
