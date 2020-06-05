from train import init_model_optim, device
from config import configs
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

result_file = "./submit.txt"

def test(config):
    model = config["model"]
    test_loader = config["test_loader"]
    test_size = config["test_size"]
    model.to(device)
    predictions = []
    for step, data in enumerate(test_loader):
        input = Variable(data).to(device)
        output = model(input)
        pred_batch = output.max(dim=1)[1].tolist()
        predictions.extend(pred_batch)
    assert len(predictions) == test_size
    open(result_file, "w+", encoding="utf-8").write("\n".join(map(str, predictions)) + "\n")

def test_main():
    test_dataset = np.load(configs["test_file"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=configs["batch_size"], shuffle=False)
    configs["test_loader"] = test_loader
    configs["model"], _ = init_model_optim(configs)
    configs["test_size"] = test_dataset.shape[0]
    test(configs)

if __name__ == "__main__":
    test_main()
    read = lambda file: list(map(int, open(file, "r+").readlines()))
    r1, r2 = read("submit.txt"), read("submit-61.txt")
    assert len(r1) == len(r2)
    print(sum([1 if r1[i] != r2[i] else 0 for i in range(len(r1))]), len(r1))