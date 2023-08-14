


'''

import glob
import os
import pickle
import time
import math
import torch
import random
import numpy as np
from scipy.spatial.distance import jaccard

from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv
from torch.autograd import Variable

if __name__ == '__main__':
    import sys

    sys.path.append('../')
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged


class Homemade(torch.nn.Module):
    def __init__(self, args, number_of_labels, number_of_edge_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.number_edge_labels = number_of_edge_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()

        # self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        # self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        # self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        self.convolution_1 = RGCNConv(self.number_labels, self.args.filters_1, self.number_edge_labels, 10)
        self.convolution_2 = RGCNConv(self.args.filters_1, self.args.filters_2, self.number_edge_labels, 10)
        self.convolution_3 = RGCNConv(self.args.filters_2, self.args.filters_3, self.number_edge_labels, 10)

        self.attention = AttentionModule(self.args)
        # self.tensor_network = TenorNetworkModule(self.args)
        # self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_first = torch.nn.Linear(32, 150)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 166)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, e_features, features, embedding_save=""):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        # convert edge index to feature ids
        edge_type = torch.tensor(np.where(e_features.numpy() == 1)[1])
        # print(edge_index.shape, edge_type, edge_type.shape)
        # print(e_features)
        features = self.convolution_1(features, edge_index, edge_type)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_2(features, edge_index, edge_type)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_3(features, edge_index, edge_type)

        # save node embeddings
        if embedding_save:
            pickle.dump(features, open(f'emb/{embedding_save}.pickle', 'wb'))

        return features

    def forward(self, data, trained=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        # edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        # features_2 = data["features_2"]
        edge_features_1 = data["e_features_1"]
        # edge_features_2 = data["e_features_2"]

        if trained:
            # print(f"Trained embeddings for {data['name_1']}, nodes: {len(data['features_1'])}")
            # print(f"Trained embeddings for {data['name_2']}, nodes: {len(data['features_2'])}")
            name_1 = data['name']
            # name_2 = data['name_2']
            abstract_features_1 = self.convolutional_pass(edge_index_1, edge_features_1, features_1,
                                                          embedding_save=name_1)
            # abstract_features_2 = self.convolutional_pass(edge_index_2, edge_features_2, features_2,
            # embedding_save=name_2)
        else:
            abstract_features_1 = self.convolutional_pass(edge_index_1, edge_features_1, features_1)
            # abstract_features_2 = self.convolutional_pass(edge_index_2, edge_features_2, features_2)

        # if self.args.histogram == True:
        # hist =self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        # pooled_features_2 = self.attention(abstract_features_2)
        # scores = self.tensor_network(pooled_features_1, pooled_features_2)
        # scores = torch.t(scores)
        pooled_features_1 = torch.t(pooled_features_1)
        # if self.args.histogram == True:
        # scores = torch.cat((scores,hist),dim=1).view(1,-1)

        # scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_first(pooled_features_1))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, number_of_labels, number_of_edge_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.number_edge_labels = number_of_edge_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()

        # self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        # self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        # self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        self.convolution_1 = RGCNConv(self.number_labels, self.args.filters_1, self.number_edge_labels, 10)
        self.convolution_2 = RGCNConv(self.args.filters_1, self.args.filters_2, self.number_edge_labels, 10)
        self.convolution_3 = RGCNConv(self.args.filters_2, self.args.filters_3, self.number_edge_labels, 10)

        self.attention = AttentionModule(self.args)
        # self.tensor_network = TenorNetworkModule(self.args)
        # self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.fully_connected_first = torch.nn.Linear(32, 150)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 166)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, e_features, features, embedding_save=""):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        # convert edge index to feature ids
        edge_type = torch.tensor(np.where(e_features.numpy() == 1)[1])
        # print(edge_index.shape, edge_type, edge_type.shape)
        # print(e_features)
        features = self.convolution_1(features, edge_index, edge_type)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_2(features, edge_index, edge_type)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features, training=self.training)
        features = self.convolution_3(features, edge_index, edge_type)

        # save node embeddings
        if embedding_save:
            pickle.dump(features, open(f'emb/{embedding_save}.pickle', 'wb'))

        return features

    def forward(self, data, trained=False):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        # edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        # features_2 = data["features_2"]
        edge_features_1 = data["e_features_1"]
        # edge_features_2 = data["e_features_2"]

        if trained:
            # print(f"Trained embeddings for {data['name_1']}, nodes: {len(data['features_1'])}")
            # print(f"Trained embeddings for {data['name_2']}, nodes: {len(data['features_2'])}")
            name_1 = data['name']
            # name_2 = data['name_2']
            abstract_features_1 = self.convolutional_pass(edge_index_1, edge_features_1, features_1,
                                                          embedding_save=name_1)
            # abstract_features_2 = self.convolutional_pass(edge_index_2, edge_features_2, features_2,
            # embedding_save=name_2)
        else:
            abstract_features_1 = self.convolutional_pass(edge_index_1, edge_features_1, features_1)
            # abstract_features_2 = self.convolutional_pass(edge_index_2, edge_features_2, features_2)

        # if self.args.histogram == True:
        # hist =self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        # pooled_features_2 = self.attention(abstract_features_2)
        # scores = self.tensor_network(pooled_features_1, pooled_features_2)
        # scores = torch.t(scores)
        pooled_features_1 = torch.t(pooled_features_1)
        # if self.args.histogram == True:
        # scores = torch.cat((scores,hist),dim=1).view(1,-1)

        # scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_first(pooled_features_1))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score


class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels, self.number_of_edge_labels)

    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()

        # edges
        self.global_edge_labels = set()

        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels"]))
            # self.global_labels = self.global_labels.union(set(data["labels_2"]))

            # edges
            self.global_edge_labels = self.global_edge_labels.union(set(data["e_labels"]))
            # self.global_edge_labels = self.global_edge_labels.union(set(data["e_labels_2"]))

        self.global_labels = list(self.global_labels)
        self.global_edge_labels = list(self.global_edge_labels)

        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}

        # edges
        self.global_edge_labels = {val: index for index, val in enumerate(self.global_edge_labels)}

        # some edge labels are missing here not sure why..
        # hard coded 19 labels for now

        self.global_edge_labels = {str(val): index for index, val in enumerate(range(19))}

        self.number_of_labels = len(self.global_labels)
        self.number_of_edge_labels = len(self.global_edge_labels)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = [self.training_graphs[graph:graph + self.args.batch_size] for graph in
                   range(0, len(self.training_graphs), self.args.batch_size)]
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()
        edges_1 = torch.from_numpy(np.array(data["graph"], dtype=np.int64).T).type(torch.long)
        # edges_2 = torch.from_numpy(np.array(data["graph_2"], dtype=np.int64).T).type(torch.long)
        # print("EDGES")
        # print(edges_1)
        # print(edges_2)

        features_1 = torch.FloatTensor(np.array(
            [[1.0 if int(self.global_labels[node]) == int(label) else 0 for label in self.global_labels] for node in
             data["labels"]]))
        # features_2 = torch.FloatTensor(np.array([[ 1.0 if int(self.global_labels[node]) == int(label) else 0 for label in self.global_labels] for node in data["labels_2"]]))

        ### EDGE FEATURES ###
        edge_features_1 = torch.FloatTensor(np.array(
            [[1.0 if int(self.global_edge_labels[edge]) == int(label) else 0 for label in self.global_edge_labels] for
             edge in data["e_labels"]]))
        # edge_features_2 = torch.FloatTensor(np.array([[ 1.0 if int(self.global_edge_labels[edge]) == int(label) else 0 for label in self.global_edge_labels] for edge in data["e_labels_2"]]))

        # print("FEATURES")
        # print(features_1)
        # print(features_2)
        new_data["edge_index_1"] = edges_1
        # new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        # new_data["features_2"] = features_2

        # EDGE
        new_data["e_features_1"] = edge_features_1
        # new_data["e_features_2"] = edge_features_2

        # graph names
        new_data['name'] = data['name']
        # new_data['name_2'] = data['name_2']

        # normalized_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        # new_data["target"] =  torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()
        # new_data["target"] = torch.from_numpy(np.array([data["fp"]]))
        new_data["target"] = torch.tensor([data["fp"]], dtype=torch.float)
        # new_data["target"] =  torch.from_numpy(np.exp(-normalized_ged).reshape(1,1)).view(-1).float()
        return new_data

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        loser = torch.nn.BCELoss()
        dists = []
        for graph_pair in batch:
            data = process_pair(graph_pair)
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            # print(prediction)
            label = np.where(prediction.detach().numpy() > 0.5, 1, 0)
            d = jaccard(label[0], data['target'].numpy()[0])
            if d < 0.5:
                print("-" * 20)
                print(data['name'], d)
                print(f"TRUE: {data['target'].numpy()[0]}")
                print(f"PRED: {label[0]}")
                print("-" * 20)
            dists.append(d)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction)
            # t = Variable(target, requires_grad=False)
            # print(t)
            # losses = losses + torch.nn.functional.binary_cross_entropy(t, prediction)
            losses = losses + loser(prediction, data['target'])
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss, np.mean(dists), min(dists)

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        # main_index = 0 # i think this should be inside the epoch loop
        for epoch in epochs:
            ds = []
            mins = []
            main_index = 0
            batches = self.create_batches()
            self.loss_sum = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score, d, m = self.process_batch(batch)
                ds.append(d)
                mins.append(m)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score
            loss = self.loss_sum / main_index
            print(f"SUP: {np.mean(ds), min(mins)}")
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

    def unseens(self, graph_dir):
        preds = []
        for g in sorted(os.listdir(graph_dir)):
            data = process_pair(os.path.join(graph_dir, g))
            data = self.transfer_to_torch(data)
            prediction = self.model(data, trained=True)
            label = np.where(prediction.detach().numpy() > 0.5, 1, 0)
            preds.append((g, label))
        pickle.dump(preds, open('results/preds.pickle', 'wb'))
        print("unbounds dumped")

    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            # self.ground_truth.append(calculate_normalized_ged(data))
            data = self.transfer_to_torch(data)
            target = self.model(data)
            prediction = self.model(data, trained=True)
            self.scores.append(calculate_loss(prediction, target))
        # self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        n6orm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n - norm_ged_mean) ** 2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " + str(round(base_error, 5)) + ".")
        print("\nModel test error: " + str(round(model_error, 5)) + ".")
'''


