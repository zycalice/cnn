import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from torch import nn
import collections


def load_data(file_path, reshape_images):
    df = pd.read_csv(file_path)
    label = df['label'].values
    features = df.drop('label', 1).values
    if reshape_images:
        features = features.reshape(-1, 1, 28, 28)
    return features, label


############################################################
# Neural Networks
############################################################

class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x)


# helper function 1
def fc_bn_relu(input_dim, output_dim):
    # use ReLu before bn
    return nn.Sequential(collections.OrderedDict([
        ("linear", nn.Linear(input_dim, output_dim)),
        ("relu", nn.ReLU()),
        ("bn", nn.BatchNorm1d(output_dim))
    ]))


class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        # input and output sizes
        self.m = nn.Sequential(collections.OrderedDict([
            ("hidden1", fc_bn_relu(28 * 28, 500)),
            ("hidden2", fc_bn_relu(500, 80)),
            ("output_layer", nn.Linear(80, 10))
        ]))

    def forward(self, x):
        return self.m(x)


# helper function 2
def conv_bn(ci, co, ksz, s=1, pz=0):
    return nn.Sequential(collections.OrderedDict([
        ("conv", nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=ksz, stride=s, padding=pz)),
        ("relu", nn.ReLU(True)),
        ("bn", nn.BatchNorm2d(co))
    ]))


class AdvancedModel(torch.nn.Module):
    # input data is 2d in this case
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.m = nn.Sequential(collections.OrderedDict([
            ("dropout", nn.Dropout(0.5)),
            ("conv1", conv_bn(1, 24, 5, 1, 1)),
            ("conv2", conv_bn(24, 72, 5, 1, 1)),
            ("avg_pool", nn.AvgPool2d(4)),
            ("flatten", nn.Flatten()),
            ("fc1", fc_bn_relu(2592, 512)),
            ("fc2", fc_bn_relu(512, 28)),
            ("output_layer", nn.Linear(28, 10))
        ]))

    def forward(self, x):
        return self.m(x)


############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i + 1}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100. * accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100. * f1_score(y_true, y_predicted, average="weighted"):.4f}')


def evaluate(model, data_loader):
    """
    evaluate on all the train data
    TODO: perform train val test split and train only on train and validate on val
    :param model:
    :param data_loader:
    :return:
    """
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted


def evaluation_scores(true, pred, model_name):
    accuracy = 100. * accuracy_score(true, pred)
    f1 = 100. * f1_score(true, pred, average="weighted")

    print(f'{model_name}: '
          f'Final Train Accuracy: {accuracy:.4f},',
          f'Final Train F1 Score: {f1:.4f}')


def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(evaluation_scores(y_true_easy, y_pred_easy, "Easy Model"))
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(evaluation_scores(y_true_medium, y_pred_medium, "Medium Model"))
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(evaluation_scores(y_true_advanced, y_pred_advanced, "Advanced Model"))
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')


if __name__ == '__main__':
    main()
