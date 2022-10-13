import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from utils import seed_everything
from evaluation import hierarchical_error
from dataset import ImageFolderNotAlphabetic
from tree import get_tree_from_file

from sklearn.metrics import confusion_matrix, classification_report
from transformers import ViTForImageClassification
import argparse
import os


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("-hl", "--hloss", required=True, help="Using loss hierarchical or not")
    ap.add_argument("-m", "--model", required=True, help="Inception, ResNet or ViT")
    ap.add_argument("-d", "--dataset", required=True, help="imagenet, fgvc, cifar, bones")
    ap.add_argument("-tp", "--tree", required=True, help="Path to the tree file")
    ap.add_argument("-dp", "--dataset_path", required=True,
                    help="Path to the dataset folder containing train and test folders")
    ap.add_argument("-b", "--batch_size", required=False, default=64, help="Batch size")
    ap.add_argument("-op", "--model_path", required=True, help="Path to model")

    args = vars(ap.parse_args())

    architecture = args["model"]
    dataset = args["dataset"]
    tree_file = args["tree"]
    dataset_path = args["dataset_path"]
    model_name = args["model_path"]
    batch_size = int(args["batch_size"])

    hierarchical_loss = (args["hloss"] == "True")
    regularization = (args["hloss"] == "True")

    dict_architectures = {"inception": 299, "resnet": 224, "vit": 224}
    image_size = dict_architectures[architecture]

    test_dir = os.path.join(dataset_path, "test")

    tree = get_tree_from_file(tree_file)

    all_leaves = [leaf.name for leaf in tree.leaves]

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    test_dataset = ImageFolderNotAlphabetic(test_dir, classes=all_leaves, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
    model.to(device)
    model.eval()

    y_pred = []
    y_true = []

    # iterate over test data
    h_err = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).logits if architecture == "vit" else model(inputs)

        h_err += hierarchical_error(outputs, labels, tree, all_leaves, device)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    print(f"Hierarchical error is {h_err/dataset_size:.4f}")

    # 2) Confusion Matrixes

    # 2.1) CLASSES
    print(classification_report(y_true, y_pred))
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)

