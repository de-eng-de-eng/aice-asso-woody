import os

import numpy as np
from flask import Flask
from aicentro.loader.tensorflow_loader import TensorflowLoader
from aicentro.serving.base_serving import BaseServing

class IrisServing(BaseServing):

    def __init__(self, loader, inputs_key='input', outputs={'dense_1/Softmax:0':'score'}, labels=None):
        super().__init__(loader, inputs_key=inputs_key, outputs=outputs)
        self.labels = labels

    def post_processing(self, response):
        resp_dict=dict()
        resp_dict['score']= response['dense_3/Softmax:0'].tolist()

        if self.labels:
            label_idx=np.argmax(response['dense_3/Softmax:0'], axis=1).reshape(-1,1)
            resp_dict['label']=[[self.labels[i] for i in idx] for idx in label_idx]

        return resp_dict

    def get(self, hash_url):
        metadata = self.loader.print_model_metadata()
        return metadata

    def post(self, hash_url):
        outputs_dict = self.loader.predict(self.inputs_key, self.inputs)
        return outputs_dict


app = Flask(__name__)
tensorflow_loader = TensorflowLoader()
tensorflow_loader.print_model_metadata()

baseServing = IrisServing.as_view('serving', tensorflow_loader, labels=['n', 'y'])
app.add_url_rule('/<hash_url>/', view_func=baseServing, methods=['GET', 'POST'])

if __name__ =='__main__':
    app.run(host='0.0.0.0')
