import numpy as np
from dataset import NICODataset
from torch.utils.data import DataLoader
from config import configs
from dataset import split_by_context, split_random
from train import init_model_optim, train
from baselines import sml_train
from visualization import visualize
from mlp import MLP
import torch.optim

load_dataset = lambda config: np.load(config["dataset_file"])

def main():
    raw_dataset = load_dataset(configs)
    # for i in range(7):
    train_set, valid_set = split_by_context(raw_dataset, configs)
    # train_set, valid_set = split_random(raw_dataset, configs)
    train_set = NICODataset(train_set)
    valid_set = NICODataset(valid_set)

    # feature_max = np.maximum(np.max(train_set.input, axis=0), np.max(valid_set.input, axis=0))
    # train_set.feature_max = valid_set.feature_max = feature_max
    # train_set.binarize()
    # valid_set.binarize()
    # configs["batch_size"] = len(train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=configs["batch_size"], shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=configs["batch_size"], shuffle=False)
    model, optim = init_model_optim(configs)
    configs["train_loader"] = train_loader
    configs["valid_loader"] = valid_loader
    configs["model"] = model
    configs["optim"] = optim
    print("Dataset and Model Initiate Done")
    train(configs)

def baseline_main():
    raw_dataset = load_dataset(configs)
    train_set, valid_set = split_random(raw_dataset, configs)
    # train_set, valid_set = split_by_context(raw_dataset, configs)
    configs["raw_train_set"] = train_set
    configs["raw_valid_set"] = valid_set
    sml_train(configs)

def visualize_main():
    raw_dataset = load_dataset(configs)
    visualize(raw_dataset)

if __name__ == "__main__":
    main()
    # baseline_main()
    # visualize_main()