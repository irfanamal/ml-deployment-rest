import numpy as np
import torch

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from model import BertForChainClassification1, BertForChainClassification2
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

app = Flask(__name__)
api = Api(app)

cat1_model = AutoModelForSequenceClassification.from_pretrained(
    "./bert-base-uncased-classification-chain-1"
)

cat2_model = BertForChainClassification1.from_pretrained(
    "./bert-base-uncased-classification-chain-2"
)

cat3_model = BertForChainClassification2.from_pretrained(
    "./bert-base-uncased-classification-chain-3"
)

parser = reqparse.RequestParser()
parser.add_argument('query', location='args')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class PredictCategory(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # vectorize the user's query and make a prediction
        tokenized_input = [tokenizer(user_query).input_ids]
        tokenized_input = torch.tensor(tokenized_input)
        pred = cat1_model(tokenized_input)
        cat1 = np.argmax(pred.logits.detach().numpy(), axis=1)

        pred = cat2_model(tokenized_input, cat1=torch.tensor(cat1))
        cat2 = np.argmax(pred.logits.detach().numpy(), axis=1)

        pred = cat3_model(tokenized_input, cat1=torch.tensor(cat1), cat2=torch.tensor(cat2))
        cat3 = np.argmax(pred.logits.detach().numpy(), axis=1)
        
        cat1 = cat1_model.config.id2label[cat1[0]]
        cat2 = cat2_model.config.id2label[cat2[0]]
        cat3 = cat3_model.config.id2label[cat3[0]]
        
        output = {'Cat1': cat1, 'Cat2': cat2, 'Cat3': cat3}
        
        return output

api.add_resource(PredictCategory, '/')

if __name__ == '__main__':
    app.run(debug=True)