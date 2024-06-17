from dataset import SemEval2016
from features import AnoVa_V1

dataset = SemEval2016()
train_data, test_data = dataset.get_data("train"), dataset.get_data("test")

anova = AnoVa_V1(train_data)
anova.get_anova_top_words()
