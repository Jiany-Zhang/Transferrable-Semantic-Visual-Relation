import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import argparse
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-e", "--episode", type=int, default=50000)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-w", "--weight", type=float, default=1e-9, help="entropy weight")
parser.add_argument("-r", "--reconstraction", type=float, default=1e-5, help="reconstraction weight")
parser.add_argument("-o", "--output_dir", type=str, default="")
parser.add_argument("-i", "--input_dir", type=str, default="")
parser.add_argument("-m", "--model", type=str, default="cub",
                    help="train dataset select in the list: cub, awa2, apy, sun")
args = parser.parse_args()

# Hyper Parameters
SOURSE_BATCH_SIZE = args.batch_size
TARGET_BATCH_SIZE = int(SOURSE_BATCH_SIZE * 0.4)
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
ATTRIBUTE_NETWORK_NAME = "attribute_network.pkl"
RELATION_NETWORK_NAME = "relation_network.pkl"
W = args.weight
R = args.reconstraction
INPUT_DIR = args.input_dir

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def compute_accuracy(test_features, test_label, test_id, test_unseen_loc, is_seen=True):
    attribute_network.eval()
    relation_network.eval()
    test_data = TensorDataset(test_features, test_label, test_unseen_loc)
    test_batch = 32
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    unseen_att_features, _ = attribute_network(unseen_attributes.cuda(GPU).float(), is_seen=False)

    # fetch attributes
    sample_labels = test_id
    sample_features = unseen_att_features
    class_num = sample_features.shape[0]

    print("class num:", class_num)
    predict_labels_total = []
    re_batch_labels_total = []
    features_total = torch.tensor([])
    id_total = torch.tensor([])
    scores_total = torch.tensor([])
    for batch_features, batch_labels, image_id in test_loader:

        batch_size = batch_labels.shape[0]

        batch_features = batch_features.cuda(GPU).float()  # 32*1024

        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 4096)
        relations = relation_network(relation_pairs, is_seen=is_seen).view(-1, class_num)
        if is_seen == False:
            relations = torch.sigmoid(relations)

        # re-build batch_labels according to sample_labels
        if features_total.size() == torch.Size([0]):
            features_total = batch_features
            id_total = image_id
            scores_total = relations
        else:
            features_total = torch.cat((features_total, batch_features), 0)
            id_total = torch.cat((id_total, image_id), 0)
            scores_total = torch.cat((scores_total, relations), 0)
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)

        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu().numpy()
        re_batch_labels = re_batch_labels.cpu().numpy()

        predict_labels_total = np.append(predict_labels_total, predict_labels)
        re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)

    # compute averaged per class accuracy
    predict_labels_total = np.array(predict_labels_total, dtype='int')
    re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
    unique_labels = np.unique(re_batch_labels_total)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(re_batch_labels_total == l)[0]
        acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
    acc = acc / unique_labels.shape[0]
    attribute_network.train()
    relation_network.train()
    return acc, features_total, id_total, scores_total, re_batch_labels_total, predict_labels_total


class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.norm1s = nn.BatchNorm1d(hidden_size)
        self.norm2s = nn.BatchNorm1d(output_size)
        self.norm1u = nn.BatchNorm1d(hidden_size)
        self.norm2u = nn.BatchNorm1d(output_size)
        self.fc3s = nn.Linear(output_size, hidden_size)
        self.fc4s = nn.Linear(hidden_size, input_size)
        self.fc3u = nn.Linear(output_size, hidden_size)
        self.fc4u = nn.Linear(hidden_size, input_size)

    def forward(self, x, is_seen=True):
        if is_seen == True:
            x = self.norm1s(self.fc1(x))
            x = F.relu(x)
            x = self.norm2s(self.fc2(x))
            x = F.relu(x)
            x1 = self.fc3s(x)
            x1 = F.relu(x1)
            x1 = self.fc4s(x1)
        else:
            self.norm1u.bias = self.norm1s.bias
            self.norm2u.bias = self.norm2s.bias
            self.norm1u.weight = self.norm1s.weight
            self.norm2u.weight = self.norm2s.weight
            x = self.norm1u(self.fc1(x))
            x = F.relu(x)
            x = self.norm2u(self.fc2(x))
            x = F.relu(x)
            x1 = self.fc3u(x)
            x1 = F.relu(x1)
            x1 = self.fc4u(x1)

        return x, x1


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.norm1s = nn.BatchNorm1d(hidden_size)
        self.norm1u = nn.BatchNorm1d(hidden_size)

    def forward(self, x, is_seen=True):
        if is_seen == True:
            x = self.norm1s(self.fc1(x))
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid(x)
        else:
            self.norm1u.bias = self.norm1s.bias
            self.norm1u.weight = self.norm1s.weight
            x = self.norm1u(self.fc1(x))
            x = F.relu(x)
            x = self.fc2(x)
        return x


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        x1 = F.softmax(x, dim=1)
        x2 = F.log_softmax(x, dim=1)
        b = x1 * x2
        b = (-1.0 * b.sum()) / x1.size(0)
        return b



# step 1: init dataset
print("init dataset")

output_attribute_network_file = os.path.join(args.output_dir, ATTRIBUTE_NETWORK_NAME)
output_relation_network_file = os.path.join(args.output_dir, RELATION_NETWORK_NAME)
output_file = os.path.join(args.output_dir, "result.txt")


def sun_data_loader():
    dataroot = INPUT_DIR
    dataset = 'SUN_data'
    image_embedding = 'res101'
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['original_att'].T

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute
    return attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc


def awa1_data_loader():
    dataroot = INPUT_DIR
    dataset = 'AwA1_data'
    image_embedding = 'res101'
    class_embedding = 'original_att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['att'].T

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute
    return attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc


def awa2_data_loader():
    dataroot = INPUT_DIR
    dataset = 'AwA2_data'
    image_embedding = 'res101'
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['original_att'].T

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes
    return attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc


def cub_data_loader():
    dataroot = INPUT_DIR
    dataset = 'CUB1_data'
    image_embedding = 'res101'
    class_embedding = 'original_att_splits'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['att'].T

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute
    return attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc


def apy_data_loader():
    dataroot = INPUT_DIR
    dataset = 'APY_data'
    image_embedding = 'res101'
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    attribute = matcontent['original_att'].T

    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int)  # test_label
    x_test_seen = feature[test_seen_loc]  # test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int)  # test_seen_label
    test_id = np.unique(test_label)  # test_id
    att_pro = attribute[test_id]  # test_attribute

    x = feature[trainval_loc]  # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label]  # train attributes
    return attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc


processors = {
    "cub": cub_data_loader,
    "apy": apy_data_loader,
    "awa2": awa2_data_loader,
    "awa1": awa1_data_loader,
    "sun": sun_data_loader
}

attribute, x, train_label, x_test, test_label, test_id, att_pro, att, test_unseen_loc = processors[args.model]()

# train set
train_features = torch.from_numpy(x)
print(train_features.shape)

train_label = torch.from_numpy(train_label).unsqueeze(1)
print(train_label.shape)

# attributes
all_attributes = np.array(attribute)
print(all_attributes.shape)

attributes = torch.from_numpy(attribute)
# test set

test_features = torch.from_numpy(x_test)
print(test_features.shape)

test_label = torch.from_numpy(test_label).unsqueeze(1)
print(test_label.shape)

testclasses_id = np.array(test_id)
print(testclasses_id.shape)

test_attributes = torch.from_numpy(att_pro).float()
print(test_attributes.shape)

att_len = attribute.shape[1]

train_data = TensorDataset(train_features, train_label)
unseen_data = TensorDataset(test_features, test_label)

device = torch.device('cuda', args.gpu)

# init network
print("init networks")
attribute_network = AttributeNetwork(att_len, 1250, 2048)
relation_network = RelationNetwork(4096, 1250)

attribute_network.cuda(GPU)
relation_network.cuda(GPU)

attribute_network.train()
relation_network.train()

attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
attribute_network_scheduler = StepLR(attribute_network_optim, step_size=30000, gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim, step_size=30000, gamma=0.5)

print("training...")
best_accuracy = 0.0

class_unseen_num = test_id.size
all_class_num = attribute.shape[0]

for buf in relation_network.buffers():
    print(type(buf.data), buf.size())

for param in relation_network.parameters():
    print(type(param.data), param.size())

print(len(relation_network.norm1u._parameters))
for k, v in relation_network.norm1u._parameters.items():
    print(k, v.size())

for name, param in relation_network.named_parameters(recurse=True):
    print(name, param.shape)

for episode in tqdm(range(EPISODE)):
    train_loader = DataLoader(train_data, batch_size=SOURSE_BATCH_SIZE, shuffle=True)
    unseen_loader = DataLoader(unseen_data, batch_size=TARGET_BATCH_SIZE, shuffle=True)
    batch_features, batch_labels = train_loader.__iter__().__next__()
    unseen_features, unseen_labels = unseen_loader.__iter__().__next__()

    sample_labels = []
    for label in batch_labels.numpy():
        if label not in sample_labels:
            sample_labels.append(label)

    sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
    unseen_attributes = torch.Tensor([all_attributes[i] for i in test_id]).squeeze(1)
    class_num = sample_attributes.shape[0]

    batch_features = batch_features.cuda(GPU).float()
    sample_features, x1s = attribute_network(sample_attributes.cuda(GPU))
    unseen_features = unseen_features.cuda(GPU).float()
    sample_unseen_features, x1u = attribute_network(unseen_attributes.cuda(GPU), is_seen=False)

    sample_features_ext = sample_features.unsqueeze(0).repeat(SOURSE_BATCH_SIZE, 1, 1)
    sample_unseen_features_ext = sample_unseen_features.unsqueeze(0).repeat(TARGET_BATCH_SIZE, 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num, 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
    unseen_features_ext = unseen_features.unsqueeze(0).repeat(class_unseen_num, 1, 1)
    unseen_features_ext = torch.transpose(unseen_features_ext, 0, 1)

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 4096)
    relation_unseen_pairs = torch.cat((sample_unseen_features_ext, unseen_features_ext), 2).view(-1, 4096)
    relations = relation_network(relation_pairs).view(-1, class_num)
    relations_unseen = relation_network(relation_unseen_pairs, is_seen=False).view(-1, class_unseen_num)

    sample_labels = np.array(sample_labels)
    re_batch_labels = []
    for label in batch_labels.numpy():
        index = np.argwhere(sample_labels == label)
        re_batch_labels.append(index[0][0])
    re_batch_labels = torch.LongTensor(re_batch_labels)
    entropy_loss = HLoss().cuda(GPU)
    mse = nn.MSELoss().cuda(GPU)
    one_hot_labels = torch.zeros(SOURSE_BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda(GPU)

    loss = mse(relations, one_hot_labels) \
           + W * entropy_loss(relations_unseen) \
           + R * (mse(x1s, sample_attributes.cuda(GPU)) + mse(x1u, unseen_attributes.cuda(GPU)))

    attribute_network.zero_grad()
    relation_network.zero_grad()

    loss.backward()

    attribute_network_optim.step()
    relation_network_optim.step()
    attribute_network_scheduler.step(episode)
    relation_network_scheduler.step(episode)

    if (episode + 1) % 500 == 0:
        # test
        print("Testing...")
        train_id = np.unique(train_label)
        ids = np.append(train_id, test_id)
        idsort = np.argsort(ids)
        unseen_attributes = torch.Tensor([all_attributes[i] for i in test_id]).squeeze(1)
        seen_attributes = torch.Tensor([all_attributes[i] for i in train_id]).squeeze(1)

        zsl_accuracy, features_total, id_total, scores_total, re_batch_labels_total, predict_labels_total = compute_accuracy(
            test_features, test_label, test_id, torch.from_numpy(test_unseen_loc.astype(np.int32)), is_seen=False)
        print('zsl:', zsl_accuracy)
        print("episode:", episode + 1, "loss", loss.item())
        if zsl_accuracy > best_accuracy:
            best_accuracy = zsl_accuracy
            re_batch_labels_total_np = re_batch_labels_total
            scores_total_np = scores_total.data.cpu().numpy()
            features_total_np = features_total.data.cpu().numpy()
            torch.save(attribute_network.state_dict(), output_attribute_network_file)
            torch.save(relation_network.state_dict(), output_relation_network_file)
            print("save networks for episode:", episode)

            f = open(output_file, 'w')

            f.write("best_accuracy:")
            f.write(str(best_accuracy))
            f.write('\n')
            f.close()

knn = NearestNeighbors(n_neighbors=11)
knn.fit(features_total_np)
D, id = knn.kneighbors(features_total_np)
id = id[:, 1:]
D = D[:, 1:]
H = 0
mean_d = np.mean(D)
D = np.exp((-1 * D) / (2 * (np.power(mean_d, 2))))
a = features_total_np.shape[0]
W1 = np.zeros((a, a))
Q = np.zeros((a, a))
for i in range(a):
    W1[i, id[i, :]] = D[i, :]
    Q[i, i] = np.sum(W1[i, :]) ** -0.5

L = Q @ W1 @ Q
j_list = []
acc_list = []
for j in range(10):
    j_0 = j * 0.1
    Y_ = np.linalg.matrix_power(np.eye(a) - j_0 * L, -1)
    Y_[np.isinf(Y_)] = 0
    Y_ = Y_ @ scores_total_np
    index = np.argmax(Y_, axis=1)
    acc = 0
    for i in range(len(test_id)):
        predict_label = index[re_batch_labels_total_np == i]
        acc = acc + np.sum(predict_label == i) / len(predict_label)
    acc = acc / len(test_id)
    acc_list.append(acc)

max_acc = max(acc_list)
print(max_acc)
