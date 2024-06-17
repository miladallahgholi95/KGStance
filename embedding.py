import torch
from transformers import BertTokenizer, BertModel

class BERT:
    def __init__(self, bert_model='bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = 'bert-base-multilingual-cased'

    def get_text_bert_embedding(self, input_text):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertModel.from_pretrained(self.model_name).to(self.device)
        tokens = tokenizer.encode_plus(input_text, add_special_tokens=True, padding='max_length', truncation=True,
                                       max_length=512, return_tensors='pt').to(self.device)
        outputs = model(**tokens)
        word_embeddings = outputs.last_hidden_state
        text_embedding = torch.mean(word_embeddings, dim=1).flatten().tolist()
        return text_embedding


a = BERT().get_text_bert_embedding(["hi my name is milad.", "how are you today?"])
print(a)