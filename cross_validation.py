import numpy as np
import pandas as pd


class CrossValidation:

    def __init__(self, data, chunk_count, dep_var, ind_vars, init_weights):
        self.data = data
        self.chunk_count = chunk_count
        self.dep_var = dep_var
        self.ind_vars = ind_vars
        self.init_weights = init_weights
        self.chunks = []
        self.traintest_pairs = []
        self.test_info = []
        self.results = None

    def split_data(self):
        n = len(self.data)
        chunk_size = round(n / self.chunk_count, 0)
        for i in range(self.chunk_count-1):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            self.chunks.append(self.data.iloc[start:end])
        fin_start = int(chunk_size * len(self.chunks))
        self.chunks.append(self.data.iloc[fin_start:])
        for i in range(len(self.chunks)):
            x = self.chunks.copy()
            test = x.pop(i)
            train = pd.concat([d for d in x])
            pair = (train, test)
            self.traintest_pairs.append(pair)

    def regress_and_test(self, data_pair, l_rate, iters, verbose=1):
        y_train = np.array((data_pair[0][self.dep_var]))
        n_train = len(y_train)
        betas = len(self.ind_vars) + 1
        x_train = np.zeros((n_train, betas))
        for i in range(betas):
            if i == 0:
                x_train[:, i] = np.ones((n_train))
            else:
                x_train[:, i] = np.array(data_pair[0][self.ind_vars[i - 1]])
        w = np.array((self.init_weights))

        tick = 0
        while tick < iters:
            pds = np.zeros((w.shape))
            for i in range(betas):
                pds[i] = - sum((y_train - np.dot(x_train, w)) * x_train[:, i]) / n_train
            w -= pds * l_rate
            if (tick % 2000 == 0) and (verbose == 1):
                print('\nIteration #' + str(tick + 1))
                print('Weights:   ' + str(w))
                print('Cost:      ' + str(round(sum((y_train - np.dot(x_train, w)) ** 2) / n_train, 3)))
            tick += 1

        y_test = np.array(data_pair[1][self.dep_var])
        n_test = len(y_test)
        x_test = np.zeros((n_test, betas))
        for i in range(betas):
            if i == 0:
                x_test[:, i] = np.ones((n_test))
            else:
                x_test[:, i] = np.array(data_pair[1][self.ind_vars[i - 1]])
        test_cost = sum((y_test - np.dot(x_test, w)) ** 2) / n_test
        z = [test_cost] +  list(w)
        self.test_info.append(z)

    def results_to_csv(self, file_path):
        runs = self.chunk_count
        cols = 2 + len(self.ind_vars)
        results_cols = np.array((['test_cost', 'bias'] + self.ind_vars))
        results_data = np.zeros((runs, cols))
        for i in range(runs):
            results_data[i] = np.array((self.test_info[i]))
        self.results = pd.DataFrame(results_data, columns=results_cols)
        print('\n Results:\n' + str(self.results))
        self.results.to_csv(file_path)
