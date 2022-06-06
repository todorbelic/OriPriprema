import copy
import random

from fontTools.misc.py23 import xrange
import  numpy as np

class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # podaci koji pripadaju ovom klasteru
        self.full_data = []

    def recalculate_center(self):
        # TODO 1: implementirati racunanje centra klastera
        # centar klastera se racuna kao prosecna vrednost svih podataka u klasteru
        new_center = [0.0 for i in range(len(self.center))]
        for dot in self.data:
            for i in range(len(dot)):
                new_center[i] += dot[i]
        n = len(self.data)
        if n != 0:
            self.center = [x/n for x in new_center]

    def get_churn(self):
        count = 0
        for data in self.full_data:
            if data[-1]:
                count +=1
        return count / len(self.full_data)


class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.full_data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []

    def fit(self, data, full_data, normalize=False):
        self.data = data  # lista N-dimenzionalnih podataka
        self.full_data = full_data
        # TODO 4: normalizovati podatke pre primene k-means
        if normalize:
            self.data = self.normalize_data(self.data)

        picked_points = []
        skip = False
        while len(picked_points) < self.n_clusters:
            point = self.data[random.randint(0, len(self.data) - 1)]
            for pt in picked_points:
                if np.array_equal(pt, point):
                    skip = True
            if skip:
                skip = False
                continue
            picked_points.append(point)
            self.clusters.append(Cluster(point))
        # TODO 1: implementirati K-means algoritam za klasterizaciju podataka
        # kada algoritam zavrsi, u self.clusters treba da bude "n_clusters" klastera (tipa Cluster)
        iter_no = 0
        not_moves = False
        while iter_no <= self.max_iter and (not not_moves):
            for cluster in self.clusters:
                cluster.data = []
                cluster.full_data = []

            for i, dot in enumerate(self.data):
                cluster_index = self.predict(dot)
                self.clusters[cluster_index].data.append(dot)
                self.clusters[cluster_index].full_data.append(self.full_data[i])

            not_moves = True
            for cluster in self.clusters:
                old_center = copy.deepcopy(cluster.center)
                cluster.recalculate_center()
                not_moves = not_moves and (np.array_equal(cluster.center, old_center))
            print("Iter no:" + str(iter_no))
            iter_no += 1
        # TODO (domaci): prosiriti K-means da stane ako se u iteraciji centri klastera nisu pomerili


    def normalize_data(self, data):

        cols = len(data[0])
        for col in range(cols):
            column_data = []
            for row in data:
                column_data.append(row[col])

            mean = np.mean(column_data)
            std = np.std(column_data)

            for row in data:
                row[col] = (row[col] - mean) / std

            return data

    def predict(self, dot):
        # TODO 1: implementirati odredjivanje kom klasteru odredjeni podatak pripada
        # podatak pripada onom klasteru cijem je centru najblizi (po euklidskoj udaljenosti)
        # kao rezultat vratiti indeks klastera kojem pripada
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = self.euclidian_distance(dot, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance
        return cluster_index


    def euclidian_distance(self, x, y):
        sq_sum = 0
        for xi, yi in zip(x, y):
            sq_sum += (yi - xi) ** 2

        return sq_sum ** 0.5

    def sum_squared_error(self):
        # TODO 3: implementirati izracunavanje sume kvadratne greske
        # SSE (sum of squared error)
        # unutar svakog klastera sumirati kvadrate rastojanja izmedju podataka i centra klastera

        sse = 0
        for cluster in self.clusters:
            for dot in cluster.data:
                sse += (self.euclidian_distance(dot, cluster.center) ** 2)
        return sse