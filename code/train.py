import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

from evaluation import accuracy_coarser_classes
from losses import hierarchical_cc_treebased
from dataset import train_val_dataset, ImageFolderNotAlphabetic
from utils import decimal_to_string, seed_everything
from tree import get_tree_from_file, get_all_labels_downtop, return_matrixes_downtop, get_all_labels_topdown, return_matrixes_topdown

import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import timeit
from anytree import LevelOrderGroupIter
import argparse
from transformers import ViTForImageClassification


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("-hl", "--hloss", required=True, help="Using loss hierarchical or not")
    ap.add_argument("-m", "--model", required=True, help="Inception, ResNet or ViT")
    ap.add_argument("-d", "--dataset", required=True, help="imagenet, fgvc, cifar, bones")
    ap.add_argument("-r", "--reduction", required=True, help="Reduction factor")
    ap.add_argument("-tp", "--tree", required=True, help="Path to the tree file")
    ap.add_argument("-dp", "--dataset_path", required=True, help="Path to the dataset folder containing train and test folders")
    ap.add_argument("-op", "--output_path", required=True, help="Path to folder where to save models")
    ap.add_argument("-lp", "--log_path", required=True, help="Path to folder where to save logs")

    ap.add_argument("-b", "--batch_size", required=False, default=64, help="Batch size")
    ap.add_argument("-e", "--epochs", required=False, default=30, help="Number of epochs")
    ap.add_argument("-lr", "--learning_rate", required=False, default=0.001, help="Learning rate")
    ap.add_argument("-wd", "--weight_decay", required=False, default=0.1, help="Weight decay")
    ap.add_argument("-pm", "--plot_matrices", required=False, default=False, help="Weather to plot or not matrices")

    args = vars(ap.parse_args())

    architecture = args["model"]
    dataset = args["dataset"]

    dict_architectures = {"inception": [299], "resnet": [224], "vit": [224]}

    image_size = dict_architectures[architecture][0]
    n_epochs = int(args["batch_size"])
    learning_rate = float(args["learning_rate"])
    weight_decay = float(args["weight_decay"])
    batch_size = int(args["batch_size"])
    reduction_factor = int(args["reduction"])
    tree_file = args["tree"]
    dataset_path = args["dataset_path"]
    model_path = args["output_path"]
    logs_path = args["logs_path"]

    plot_matrices = (args["plot_matrices"] == "True")
    hierarchical_loss = (args["hloss"] == "True")
    regularization = (args["hloss"] == "True")

    validation_split = 0.1

    name = f"{architecture}-{dataset}"

    tree = get_tree_from_file(tree_file)

    all_leaves = [leaf.name for leaf in tree.leaves]

    all_nodes_names = [[node.name for node in children] for children in LevelOrderGroupIter(tree)][1:]
    all_nodes = [[node for node in children] for children in LevelOrderGroupIter(tree)][1:]

    all_labels_topdown = get_all_labels_topdown(tree)
    all_labels_downtop = get_all_labels_downtop(tree)
    all_labels = [*all_labels_topdown, *all_labels_downtop]

    matrixes_topdown = return_matrixes_topdown(tree, plot=plot_matrices)
    matrixes_downtop = return_matrixes_downtop(tree, plot=plot_matrices)
    matrixes = [*matrixes_topdown, *matrixes_downtop]

    lens = [len(set(n)) for n in all_labels]

    # Path
    if hierarchical_loss and not regularization:
        model_name = os.path.join(model_path,
                                  f"{name}_hloss_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    elif regularization and not hierarchical_loss:
        model_name = os.path.join(model_path,
                                  f"{name}_reg_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    elif regularization and hierarchical_loss:
        model_name = os.path.join(model_path,
                                  f"{name}_hloss_reg_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    else:
        model_name = os.path.join(model_path,
                                  f"{name}_lr{decimal_to_string(learning_rate)}_wd{decimal_to_string(weight_decay)}_1on{reduction_factor}.pth")
    print(f"Model name: {model_name}")

    # Log
    writer = SummaryWriter(os.path.join(logs_path, model_name.split("//")[-1].split(".")[0]))

    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), Resize((image_size, image_size))])

    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")

    # Load the data: train and test sets
    train_dataset = ImageFolderNotAlphabetic(train_dir, classes=all_leaves, transform=transform)
    dataset = train_val_dataset(train_dataset, validation_split, reduction_factor, reduce_val=False)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}

    # Check lr_ratio
    lr_ratio = 1 / len(train_loader)
    print(f"LR should be around {lr_ratio:.4f}")

    # Model
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

    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if regularization \
        else torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Parameters for training
    n_total_steps_train = len(train_loader)
    n_total_steps_val = len(val_loader)
    platoon = 0
    best_acc = 0.0
    associated_medium_acc = 0.0
    associated_coarse_acc = 0.0

    for epoch in range(n_epochs):
        start = timeit.default_timer()
        print("-" * 200)
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
                n_total_steps = n_total_steps_train
            else:
                model.eval()
                loader = val_loader
                n_total_steps = n_total_steps_val

            running_loss = 0.0
            running_corrects = 0
            running_corrects_coarser_level = [0 for i in range(len(all_labels))]
            running_loss_fine = 0.0
            running_loss_coarser_level = [0.0 for i in range(len(all_labels))]

            epoch_acc_coarser = [0 for i in range(len(all_labels))]
            epoch_loss_coarser = [0.0 for i in range(len(all_labels))]

            # Iterate over data
            for j, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                print(f"Step {j} / {len(loader)}")

                # Forward
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs).logits if architecture == "vit" else model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss, loss_dict = hierarchical_cc_treebased(outputs, labels, tree, lens, all_labels, all_leaves,
                                                                model, 0.0, device, hierarchical_loss, regularization,
                                                                weight_decay, matrixes, architecture)

                    # Backward + optimize
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
                running_loss_fine += loss_dict["loss_fine"]

                for i in range(len(all_labels)):
                    running_corrects_coarser_level[i] += accuracy_coarser_classes(outputs, labels, np.asarray(all_labels[i]),
                                                                   len(all_labels[i]), device)
                    running_loss_coarser_level[i] += loss_dict[f"loss_{i}"]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_loss_fine = running_loss_fine / dataset_sizes[phase]

            for i in range(len(all_labels)):
                epoch_acc_coarser[i] = running_corrects_coarser_level[i] / dataset_sizes[phase]
                epoch_loss_coarser[i] = running_loss_coarser_level[i] / dataset_sizes[phase]

            print(f"Step {j + 1}/{n_total_steps}, {phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            for i in range(len(all_labels)):
                print(f"{phase} Loss {i}: {epoch_loss_coarser[i]:.4f}, Accuracy {i}: {epoch_acc_coarser[i]:.4f}")

            if (j + 1) % n_total_steps == 0:
                if phase == "val":
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        platoon = 0
                        best_model_name = model_name[:-4] + "_best.pth"
                        torch.save(model.state_dict(), best_model_name)
                        print(f"New best accuracy {best_acc:.4f}, saving best model")

                    if epoch_acc < best_acc:
                        platoon += 1
                        print(f"{platoon} epochs without improvement, best accuracy: {best_acc:.4f}")

                    print("End of validation epoch.")
                    writer.add_scalar("Validation loss", epoch_loss, epoch)
                    writer.add_scalar("Validation accuracy", epoch_acc, epoch)
                    for i in range(len(all_labels)):
                        writer.add_scalar(f"Validation accuracy {i}", epoch_acc_coarser[i], epoch)

                    writer.add_scalars("Training vs. validation loss",
                                       {"Training": epoch_loss_compare, "Validation": epoch_loss}, epoch)
                    writer.add_scalars("Training vs. validation accuracy",
                                       {"Training": epoch_acc_compare, "Validation": epoch_acc}, epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_labels)):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties validation", plot_dict, epoch)

                elif phase == "train":
                    print("End of training epoch.")
                    writer.add_scalar("Training loss", epoch_loss, epoch)
                    writer.add_scalar("Training accuracy", epoch_acc, epoch)
                    for i in range(len(all_labels)):
                        writer.add_scalar(f"Training accuracy {i}", epoch_acc_coarser[i], epoch)

                    plot_dict = {"Loss": epoch_loss, "Fine Loss": epoch_loss_fine}

                    for i in range(len(all_labels)):
                        plot_dict[f"Loss {i}"] = epoch_loss_coarser[i]

                    writer.add_scalars("Losses and penalties training", plot_dict, epoch)
                    epoch_loss_compare = epoch_loss
                    epoch_acc_compare = epoch_acc

        stop = timeit.default_timer()
        print(f"Elapsed time {stop-start:.4f}")

    last_model_name = model_name[:-4] + "_last.pth"

    writer.flush()
    writer.close()
