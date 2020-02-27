from core.data.constants import LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, GRAPH
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION
from core.data.constants import GRAPH, N_RELS, N_CLASSES, N_ENTITIES
import powernet.pre_powernet as power
from powernet.power_app import power_app
from utils.io import print_graph_stats, read_params, create_default_path, remove_model



def interfece(device_id,counter_value,threshold):
    dir=power.load_node_dir()
    id=dir[device_id]
    mode = NODE_CLASSIFICATION
    data = {}
    data[GRAPH] = power.make_graph()
    data[N_CLASSES] = 2
    data[LABELS] = power.load_labels()
    data[TRAIN_MASK], data[TEST_MASK], data[VAL_MASK] = power.make_mask(674*10)
    config_params = read_params('../core/models/config_files/config_edGNN_node_class.json', verbose=True)
    pi = power_app(data, config_params[0], False)
    r1=pi.predict(id)
    r2=pi.acc(data)
    print('预测结果为：'+str(r1)+'  准确率：'+str(r2))
    return r1,r2


interfece('NE=33554500',0,20)