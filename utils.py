
from math import floor
from random import random
import numpy as np
import pandas as pd
import torch as torch
import dgl
import torch.nn.functional as F


def load_feature_data():
    # chemical_similarity as feature
    TF_chemical_feature = pd.read_csv('./data/TF_chemical_similarity_num.csv',  header=None)
    tg_chemica_feature = pd.read_csv('./data/tg_chemical_similarity_num.csv', header=None)
    go_feature = pd.read_csv('./data/go_onehot_all.csv',header=None)

    return TF_chemical_feature, tg_chemica_feature, go_feature


def sample():
    
    activation = pd.read_csv('./data/activation_new.csv',header=0)
    repression = pd.read_csv('./data/repression_new.csv',header=0)
    unknown = pd.read_csv('./data/unknown.csv',header=0)
    TF_associate_disease = pd.read_csv('./data/TF_disease_num.csv',header=0)  
    tg_associate_disease = pd.read_csv('./data/tg_disease_num.csv',header=0)
    go_associate_TF = pd.read_csv('./data/GO_TF_pairs_all.csv',header=0)  
    TF_coregulate_TF = pd.read_csv('./data/coregulate_top3_num.csv',header=0)

    return activation, repression, unknown, TF_associate_disease, tg_associate_disease, go_associate_TF, TF_coregulate_TF


def build_hetero_graph(activate_idx, repress_idx, unknown_idx, TF_disease_idx, tg_disease_idx, go_TF_idx, random_seed, device):
    
    TF_chemical_feature_origin, tg_chemical_feature_origin, go_feature_origin = load_feature_data()
    activation, repression, unknown, TF_associate_disease, tg_associate_disease, go_associate_TF, TF_coregulate_TF = sample()

    # extract TF-mode-tg rows involved in train_index of fold i 
    activation_src_node = torch.tensor(data = activation['TF'][activate_idx].values, device = device)
    activation_dst_node = torch.tensor(data = activation['tg'][activate_idx].values, device = device)
    repression_src_node = torch.tensor(data = repression['TF'][repress_idx].values, device = device)
    repression_dst_node = torch.tensor(data = repression['tg'][repress_idx].values, device = device)
    unknown_scr_node = torch.tensor(data = unknown['TF'][unknown_idx].values, device = device) 
    unknown_dst_node = torch.tensor(data = unknown['tg'][unknown_idx].values, device = device) 

    TF_disease_src_node = torch.tensor(data = TF_associate_disease['TF'][TF_disease_idx].values, device = device)
    TF_disease_dst_node = torch.tensor(data = TF_associate_disease['disease'][TF_disease_idx].values, device = device)
    tg_disease_src_node = torch.tensor(data = tg_associate_disease['tg'][tg_disease_idx].values, device = device)
    tg_disease_dst_node = torch.tensor(data = tg_associate_disease['disease'][tg_disease_idx].values, device = device)
    
    TF_disease_len = floor(len(TF_disease_src_node)/4*2)
    tg_disease_len = floor(len(tg_disease_dst_node)/4*2)

    TF_disease_src_node = TF_disease_src_node[:TF_disease_len]
    TF_disease_dst_node = TF_disease_dst_node[:TF_disease_len]
    tg_disease_src_node = tg_disease_src_node[:tg_disease_len]
    tg_disease_dst_node = tg_disease_dst_node[:tg_disease_len]

   
    TF_tg_src_node = torch.cat((activation_src_node, repression_src_node, unknown_scr_node),0)
    TF_tg_dst_node = torch.cat((activation_dst_node, repression_dst_node, unknown_dst_node),0)

    # go_TF node
    go_TF_src_node_1 = torch.tensor(data = go_associate_TF['GO'][go_TF_idx].values, device = device)
    go_TF_dst_node_1 = torch.tensor(data = go_associate_TF['TF'][go_TF_idx].values, device = device)
    
    go_TF_len = floor(len(go_TF_src_node_1)/4*2)
    go_TF_src_node_1 = go_TF_src_node_1[:go_TF_len]
    go_TF_dst_node_1 = go_TF_dst_node_1[:go_TF_len]

    # generate links between the 'NO.4337' GO node and all TF node to avoid the TF node is in-degree 
    go_TF_additional_src_list = [4337]*666
    go_TF_additional_src_node = torch.tensor(data = go_TF_additional_src_list, device = device)
    go_TF_additional_dst_node = torch.arange(0,666, device = device)

    go_TF_src_node = torch.cat((go_TF_src_node_1, go_TF_additional_src_node), 0)
    go_TF_dst_node = torch.cat((go_TF_dst_node_1, go_TF_additional_dst_node), 0)
    

    # label for link prediction
    activate_lp_label = torch.tensor(data = activation['lp_label'][activate_idx].values, device=device)
    repress_lp_label = torch.tensor(data = repression['lp_label'][repress_idx].values, device=device)
    unknown_lp_label = torch.tensor(data = unknown['lp_label'][unknown_idx].values, device=device)
    TF_tg_lp_label = torch.cat((activate_lp_label, repress_lp_label, unknown_lp_label),0)


    # label for classification
    activate_a_label = torch.tensor(data = activation['c_label_a'][activate_idx].values, device=device)
    repress_a_label = torch.tensor(data = repression['c_label_a'][repress_idx].values, device=device)
    activate_r_label = torch.tensor(data = activation['c_label_r'][activate_idx].values, device=device)
    repress_r_label = torch.tensor(data = repression['c_label_r'][repress_idx].values, device=device)
    unknown_c_label = torch.tensor(data = unknown['c_label'][unknown_idx].values, device=device)

    TF_tg_a_label = torch.cat((activate_a_label, repress_a_label, unknown_c_label),0)
    TF_tg_r_label = torch.cat((activate_r_label, repress_r_label, unknown_c_label),0)

    associate_1_label = torch.tensor(data = TF_associate_disease['lp_label'][TF_disease_idx].values, device=device)
    associate_2_label = torch.tensor(data = tg_associate_disease['lp_label'][tg_disease_idx].values, device=device)
    associate_1_label = associate_1_label[:TF_disease_len]
    associate_2_label = associate_2_label[:tg_disease_len]

    # self-loop go node
    go_self_loop_node = torch.tensor(data = range(4338), device= device)   
    TF_self_loop_node = torch.tensor(data = range(666), device= device)   
    

    # build heterogenous graph
    hetero_graph = dgl.heterograph({
        ('TF','regulate','tg') : (TF_tg_src_node, TF_tg_dst_node),
        #('TF','co_regulate','TF') : (TF_src_node, TF_dst_node),
        ('TF','associate_1','disease') : (TF_disease_src_node, TF_disease_dst_node),
        ('disease','associate_2','tg') : (tg_disease_dst_node, tg_disease_src_node),
        ('go', 'associate_3','TF') : (go_TF_src_node, go_TF_dst_node),
        ('go','go_self_loop','go'): (go_self_loop_node, go_self_loop_node),
        ('TF','TF_self_loop','TF'): (TF_self_loop_node, TF_self_loop_node),},
        idtype = torch.int32, device = device)
    
    

    # extract features involved in train_index of fold i 
    TF_ids = hetero_graph.nodes('TF').cpu().numpy().tolist()
    tg_ids = hetero_graph.nodes('tg').cpu().numpy().tolist()
    go_ids = hetero_graph.nodes('go').cpu().numpy().tolist()
    TF_chemical_feature = TF_chemical_feature_origin.iloc[TF_ids]
    tg_chemical_feature = tg_chemical_feature_origin.iloc[tg_ids]
    go_feature = go_feature_origin.iloc[go_ids]

    # generate random feature for disease node
    # random normal
    disease_feature = np.random.normal(0.5, 0.5, (hetero_graph.num_nodes('disease'), TF_chemical_feature.shape[1]))

    #co_regulate_lp_label = torch.ones(len(TF_src_node), device = device)
    associate_3_label = torch.ones(len(go_TF_src_node), device = device)

    # add nodes feature
    hetero_graph.nodes['TF'].data['feature'] = torch.as_tensor(TF_chemical_feature.values, dtype = torch.float32, device = device)
    hetero_graph.nodes['tg'].data['feature'] = torch.as_tensor(tg_chemical_feature.values, dtype = torch.float32, device = device)
    hetero_graph.nodes['disease'].data['feature'] = torch.as_tensor(disease_feature, dtype = torch.float32, device = device)
    hetero_graph.nodes['go'].data['feature'] = torch.as_tensor(go_feature.values, dtype = torch.float32, device = device)
    
    # add edges label
    hetero_graph.edges['regulate'].data['lp_label'] = TF_tg_lp_label 
    hetero_graph.edges['regulate'].data['c_label_a'] = TF_tg_a_label
    hetero_graph.edges['regulate'].data['c_label_r'] = TF_tg_r_label
    
    #hetero_graph.edges['co_regulate'].data['lp_label'] = co_regulate_lp_label
    hetero_graph.edges['associate_1'].data['lp_label'] = associate_1_label
    hetero_graph.edges['associate_2'].data['lp_label'] = associate_2_label
    hetero_graph.edges['associate_3'].data['lp_label'] = associate_3_label

    hetero_graph.edges['go_self_loop'].data['lp_label'] = torch.ones(len(go_self_loop_node), device = device)
    hetero_graph.edges['TF_self_loop'].data['lp_label'] = torch.ones(len(TF_self_loop_node), device = device)
    
    return hetero_graph, TF_chemical_feature.shape[1] 



def build_graph_for_classify(graph, lp_score, device):   

    c_label_a = graph.edges[('TF','regulate','tg')].data['c_label_a']
    c_label_r = graph.edges[('TF','regulate','tg')].data['c_label_r']

    src_nodes, dst_nodes = graph.edges(etype='regulate')
   
    lp_score = lp_score[:len(c_label_a)]
    
    ActivateLabel_of_activation = []
    RepressLabel_of_activation = []
    ActivateLabel_of_repression = []
    RepressLabel_of_repression = []

    c_activation_src_node = []
    c_activation_dst_node = []
    c_repression_src_node = []
    c_repression_dst_node = []

    for idx in range(len(lp_score)):
        if(lp_score[idx] > 0.5):
            if(c_label_a[idx] == 1):
                c_activation_src_node.append(src_nodes[idx])
                c_activation_dst_node.append(dst_nodes[idx])
                ActivateLabel_of_activation.append(1)
                if c_label_r[idx] == 1:
                    RepressLabel_of_activation.append(1)
                else:
                    RepressLabel_of_activation.append(0)
                
            elif(c_label_a[idx] == 0):    
                c_repression_src_node.append(src_nodes[idx])
                c_repression_dst_node.append(dst_nodes[idx])
                RepressLabel_of_repression.append(1)
                ActivateLabel_of_repression.append(0)


    c_graph = dgl.heterograph({
    ('TF','activate','tg'): (c_activation_src_node, c_activation_dst_node),
    ('TF','repress','tg'): (c_repression_src_node, c_repression_dst_node),
    ('tg','activate_feedback','TF'): (c_activation_dst_node, c_activation_src_node),
    ('tg','repress_feedback','TF'): (c_repression_dst_node, c_repression_src_node)
    },idtype = torch.int32, device = device)
    
    
    c_src_node_list = c_graph.nodes('TF')
    c_dst_node_list = c_graph.nodes('tg')
    

    node_sub_graph = dgl.node_subgraph(graph, {'TF': c_src_node_list, 'tg': c_dst_node_list})
    node_types = node_sub_graph.ntypes
    h = {node_types[j]: node_sub_graph.nodes[node_types[j]].data['h'] for j in range(len(node_types))}

    activate_label = ActivateLabel_of_activation + ActivateLabel_of_repression
    repress_label = RepressLabel_of_activation + RepressLabel_of_repression
    c_label = [[activate_label[i], repress_label[i]] for i in range(len(activate_label))]

    c_label = torch.tensor(data = c_label, device = device)
    
    
    return c_graph, h, c_label


def construct_negative_graph(graph, k, etypes, device):
    # only predict the ('TF' - 'activate+repress' - 'tg')etype
    # so the graph is a dec_graph

    utype, _, vtype = etypes
    src, dst = graph.edges(etype=etypes)

    neg_src = src.repeat_interleave(1).to(torch.int32).to(device)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * 1,)).to(torch.int32).to(device)

    return dgl.heterograph(
        {etypes: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

    

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if random_seed == 0:
        torch.backends.cudann.deterministic = True
        torch.backends.cudnn.benchmark = False
        








