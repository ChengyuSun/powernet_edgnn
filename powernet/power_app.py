import torch
import numpy as np
from core.data.constants import LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, GRAPH
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION
from core.data.constants import GRAPH, N_RELS, N_CLASSES, N_ENTITIES
from utils.io import load_checkpoint
from core.models.model import Model

class power_app:

    def __init__(self,data, model_config, is_cuda, mode=NODE_CLASSIFICATION):
        self.model = Model(g=data[GRAPH],
                           config_params=model_config,
                           n_classes=data[N_CLASSES],
                           n_rels=data[N_RELS] if N_RELS in data else None,
                           n_entities=data[N_ENTITIES] if N_ENTITIES in data else None,
                           is_cuda=is_cuda,
                           mode=mode)
    def predict(self, input, load_path='../bin/checkpoint_ba61d44d-ddd5-4ff4-aa0f-02730155a017.pt', mode=NODE_CLASSIFICATION):
        try:
            print('*** Load pre-trained model ***')
            self.model = load_checkpoint(self.model, load_path)
        except ValueError as e:
            print('Error while loading the model.', e)

        res=self.model.power_out(input)
        if res[1]>res[0]:
            return 1
        else:return 0


    def acc(self,data):
        test_mask = data[TEST_MASK]
        labels = data[LABELS]
        acc, _ = self.model.eval_node_classification(labels, test_mask)
        return acc
