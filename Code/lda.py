from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle

from utils import seed_everything
from dataset import ImageFolderNotAlphabetic
from tree import get_tree_from_file

from anytree import LevelOrderGroupIter
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification


def extract_features(numbers):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    batch_size = 1
    architecture = "vit"
    dataset = "cifar"
    model_name = f"..//..//Models//Mat_version_210622//{architecture}-{dataset}//{architecture}-{dataset}_hloss_reg_lr0001_wd01_1on8_best.pth"

    dict_architectures = {"inception": 299, "resnet": 224, "vit": 224}

    image_size = dict_architectures[architecture]

    test_dir = f"..//..//Dataset//{dataset}//test//"
    tree_file = f"..//..//Dataset//{dataset}//tree.txt"

    tree = get_tree_from_file(tree_file)

    all_leaves = [leaf.name for leaf in tree.leaves]

    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]

    lens = [len(n) for n in all_nodes]

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=all_leaves, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    dataset_size = len(test_loader)

    if architecture == "inception":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))
    elif architecture == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_features=len(all_leaves))
    elif architecture == "vit":
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, out_features=len(all_leaves))

    model.load_state_dict(torch.load(model_name))
    model.fc = nn.Sequential()
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    h_err = 0.0

    i = 0
    for inputs, labels in test_loader:
        # if i == 5: break
        i += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        if labels.item() in numbers:

            outputs = model(inputs).logits if architecture == "vit" else model(inputs)
            outputs = outputs.data.cpu().numpy()
            y_pred.extend(outputs)
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

    np.save("representation_hl", np.array(y_pred))
    np.save("labels_hl", np.array(y_true))

    return all_leaves


if __name__ == "__main__":

    labels = [5, 6, 7, 8, 9, 65, 66, 67, 68, 69]
    target_names = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout', 'clock', 'keyboard', 'lamp', 'telephone', 'television']

    extract_features(labels)

    X = np.load("representation_hl.npy")
    y = np.load("labels_hl.npy")
    X, y = shuffle(X, y, random_state=0)

    n_samples = X.shape[0]

    # Create a classifier: a Fisher's LDA classifier
    lda = LinearDiscriminantAnalysis(n_components=2, solver='eigen', shrinkage=0.1)

    # Train lda on the first half of the digits
    lda = lda.fit(X[:n_samples // 2], y[:n_samples // 2])
    X_r_lda = lda.transform(X)

    X_r_lda = (X_r_lda - np.min(X_r_lda)) / (np.max(X_r_lda) - np.min(X_r_lda))

    color_list = ["lime", "limegreen", "lightgreen", "green", "palegreen", "blue", "skyblue", "deepskyblue", "darkblue", "cornflowerblue"] # "tomato", "indianred", "firebrick", "red", "darkred"
    marker_list = ["o", "o", "o", "o", "o", "v", "v", "v", "v", "v"] #"o", "o"] #]

    with plt.style.context('seaborn-talk'):
        j = 0
        for i, target_name in zip(labels, target_names):
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.scatter(X_r_lda[y == i, 0], X_r_lda[y == i, 1], alpha=.6, c=color_list[j],
                            label=target_name, marker=marker_list[j])
            plt.xlabel('Discriminant Coordinate 1')
            plt.ylabel('Discriminant Coordinate 2')
            j += 1

        plt.legend(target_names)
        plt.tight_layout()

    plt.show()

    n_samples = len(X)

    # Predict the value of the digit on the second half:
    expected = y[n_samples // 2:]
    predicted = lda.predict(X[n_samples // 2:])

    report = metrics.classification_report(expected, predicted)
    print("Classification report:\n%s" % (report))