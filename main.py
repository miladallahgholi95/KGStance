from dataset import SemEval2016

dataset = SemEval2016()
train_data, test_data = dataset.get_data("train"), dataset.get_data("test")

print("1")