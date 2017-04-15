import numpy as np

class DataLoader():
    def __init__(self, num_features):
        self.num_features = num_features
        self.relevance_scores = []
        self.qids = []
        self.features = []
        pass

    def load_from_file(self, filenames):
        if isinstance(filenames, str):
            filenames = [filenames]

        lines_read = 0
        for filename in filenames:
            with open(filename, 'rb') as file:
                feature_vec = np.zeros(self.num_features, dtype=np.float32)
                for line in file:
                    lines_read += 1
                    if lines_read % 10000 == 0:
                        print('Lines read: ', lines_read)
                    line = line.decode("utf-8")
                    line = line.strip()
                    items = line.split()

                    # Get the query id and relevance score
                    self.qids.append(int(items[1].split(':')[1]))
                    self.relevance_scores.append(int(items[0]))

                    for item in items[2:]:
                        idx, val = item.split(':')
                        feature_vec[int(idx) - 1] = float(val)

                    self.features.append(feature_vec.copy())



filename = '/Users/phillipmortimer/PycharmProjects/IRDM_2017_group_42/data/Fold1/test.txt'

dataLoader = DataLoader(136)

dataLoader.load_from_file(filename)


debug = True
