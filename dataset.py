import pandas as pd


class SemEval2016:
    def __init__(self):
        self.train_path = "data/SemEval2016/SemEval-2016-Train.xlsx"
        self.test_path = "data/SemEval2016/SemEval-2016-Test-TaskA.xlsx"

    def get_data(self, data_type):

        if data_type == "train":
            data_file = pd.read_excel(self.train_path)
        elif data_type == "test":
            data_file = pd.read_excel(self.test_path)
        else:
            return None

        json_data = []
        for index, row in data_file.iterrows():
            row_dict = {
                "Text": row["Tweet"],
                "target": row["Target"],
                "Label": row["Stance"]
            }
            json_data.append(row_dict)

        return json_data
