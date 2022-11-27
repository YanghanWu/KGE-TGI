import sys
import torch as torch
import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn import svm
#from utils import Activation_function, LossFunction
from compare_model import SAGE_RGCN, GAT_RGCN, Cheb_RGCN, GIN_RGCN, Edge_RGCN, HeteroLinear



'''class HeteroLinear(nn.Module):
    def __init__(self, in_feats:dict, out_dim):
        super().__init__()
        
        self.layer1 = nn.Linear(in_feats['TF'], out_dim)
        self.layer2 = nn.Linear(in_feats['tg'], out_dim)
        self.layer3 = nn.Linear(in_feats['disease'], out_dim)
        self.layer4 = nn.Linear(in_feats['go'], out_dim)

    def forward(self, inputs:dict):
        h = {}
        h['TF'] = self.layer1(inputs['TF'])
        h['tg'] = self.layer2(inputs['tg'])
        h['disease'] = self.layer3(inputs['disease'])
        h['go'] = self.layer4(inputs['go'])
        
        return h'''



class GraphTGI_hetero(nn.Module):
    def __init__(self, in_features, hidden_features, slope):
        super().__init__()
        #self.sage = RGCN(in_features, hidden_features, slope)  
        self.sage = RGCN(in_features, hidden_features, slope)

    def forward(self, g, g_x):
        
        graph = self.sage(g, g_x)
        
        return graph



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, slope):
        super().__init__()
        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2))
        size1 = 1024
        size2 = 512
        size3 = 256

        '''in_feats_list = {'TF': in_feats, 'tg': in_feats, 'disease': in_feats, 'go': 4338}
        self.layer1 = nn.Linear(in_feats_list['TF'], size1)
        self.layer2 = nn.Linear(in_feats_list['tg'], size1)
        self.layer3 = nn.Linear(in_feats_list['disease'], size1)
        self.layer4 = nn.Linear(in_feats_list['go'], size1)'''
        self.Heterolinear = HeteroLinear({'TF': in_feats, 'tg': in_feats, 'disease': in_feats, 'go': 4338}, size1)
        #print(self.Heterolinear.device)
        
        #self.Heterolinear = HeteroLinear({'TF': in_feats, 'tg': in_feats, 'go': 4338}, size1)
        #self.Heterolinear = HeteroLinear({'TF': in_feats, 'tg': in_feats, 'disease': in_feats}, size1)
        #self.Heterolinear = HeteroLinear_v2([in_feats, in_feats, in_feats, 4338], size1)
        
      


        self.conv1 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size1, size2, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size1, size2, activation = self.act1),
            #'tg_self_loop': dglnn.GraphConv(in_feats, size1, activation = self.act1),
            #'d_self_loop': dglnn.GraphConv(in_feats, size1, activation = self.act1)
            #'co_regulate' : dglnn.GraphConv(size1, size2, activation = self.act1)
            #'reverse_associate_3' : dglnn.GraphConv(in_feats, size1, activation = self.act1)
            },
            aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            #'tg_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            #'d_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1)
            #'co_regulate' : dglnn.GraphConv(size2, size3, activation = self.act1)
            #'reverse_associate_3' : dglnn.GraphConv(size2, size3, activation = self.act1)
            },
            aggregate='sum')
        
        '''self.conv3 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size3, size4, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size3, size4, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size3, size4, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size3, size4, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size3, size4, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size3, size4, activation = self.act1),
            #'tg_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            #'d_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1)
            #'co_regulate' : dglnn.GraphConv(size2, size3, activation = self.act1)
            #'reverse_associate_3' : dglnn.GraphConv(size2, size3, activation = self.act1)
            },
            aggregate='sum')

        self.conv4 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size4, size5, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size4, size5, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size4, size5, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size4, size5, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size4, size5, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size4, size5, activation = self.act1),
            #'tg_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            #'d_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1)
            #'co_regulate' : dglnn.GraphConv(size2, size3, activation = self.act1)
            #'reverse_associate_3' : dglnn.GraphConv(size2, size3, activation = self.act1)
            },
            aggregate='sum')'''

  
        
    def forward(self, graph, inputs):
        # inputs = node features
        # transfer to same size
  
        h = self.Heterolinear(inputs)
        '''h = {}
        h['TF'] = self.layer1(inputs['TF'])
        h['tg'] = self.layer2(inputs['tg'])
        h['disease'] = self.layer3(inputs['disease'])
        h['go'] = self.layer4(inputs['go'])'''

        # GraphConv
        h1 = self.conv1(graph, h)
        h_mid = {k: self.act1(v) for k, v in h1.items()}     # 还可以加norm

        h2 = self.conv2(graph, h_mid)
        #h_mid = {k: self.act1(v) for k, v in h2.items()}
        
        #h3 = self.conv3(graph, h_mid)
        #h_mid = {k: self.act1(v) for k, v in h3.items()}

        #h4 = self.conv4(graph, h_mid)

        
        # h is a dict, key = 'TF', 'disease', 'tg'
        # 把计算得到的新数据保存为特征'h'
        node_types = graph.ntypes
        for i in range(len(node_types)):
            graph.apply_nodes(lambda nodes: {'h': h2[node_types[i]]}, ntype=node_types[i])
       
        
        # 由于后续需要直接取图的数据，所以要返回graph
        return graph




class GraphTGI_classifier(nn.Module):
    def __init__(self, in_features, hidden_features, slope):
        super().__init__()
        self.sage = RGCN_classify(in_features, hidden_features, slope)  
        self.classifier = Classify_MLPPredictor(32, 2)

    def forward(self, g, g_x):
        
        
        h = self.sage(g, g_x)
   
        # 要预测已有的边是什么类型，必须模糊掉边的信息

        dec_g = g['TF', : ,'tg']
        edge_label = dec_g.edata[dgl.ETYPE]
        
        edge_label = edge_label.to(torch.int64)
        score = self.classifier(dec_g, h)

        return score


# for classify
class RGCN_classify(nn.Module):
    def __init__(self, in_feats, hid_feats, slope):
        super().__init__()
        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2))
        size1 = 256
        size2 = 128
        size3 = 64
        size4 = 32

        self.conv1 = dglnn.HeteroGraphConv({
            'activate' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'repress': dglnn.GraphConv(size1, size2, activation = self.act1),
            'repress_feedback': dglnn.GraphConv(size1, size2, activation = self.act1),
            'activate_feedback': dglnn.GraphConv(size1, size2, activation = self.act1),
            },
            aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'activate'  : dglnn.GraphConv(size2, size3, activation = self.act1),
            'repress': dglnn.GraphConv(size2, size3, activation = self.act1),
            'activate_feedback': dglnn.GraphConv(size2, size3, activation = self.act1),
            'repress_feedback': dglnn.GraphConv(size2, size3, activation = self.act1),
            },
            aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            'activate' : dglnn.GraphConv(size3, size4, activation = self.act1),
            'repress': dglnn.GraphConv(size3, size4, activation = self.act1),
            'repress_feedback': dglnn.GraphConv(size3, size4, activation = self.act1),
            'activate_feedback': dglnn.GraphConv(size3, size4, activation = self.act1),
            },
            aggregate='sum')
        
            
        
    def forward(self, graph, inputs):
        # inputs = node features

        h1 = self.conv1(graph, inputs)   
        h_mid = {k: self.act1(v) for k, v in h1.items()}     

        h2 = self.conv2(graph, h_mid)
        h_mid = {k: self.act1(v) for k, v in h2.items()}

        h3 = self.conv3(graph, h_mid)
        
        # 由于后续需要直接取图的数据，所以要返回graph
        return h3
    

# dot product predictor for link prediction 
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            
            graph.ndata['link_prediction_h'] = h    
            graph.apply_edges(dgl.function.u_dot_v('link_prediction_h', 'link_prediction_h', 'link_prediction_score'), etype=etype)
            #return graph.edges[etype].data['link_prediction_score']
            
            return torch.sigmoid(graph.edges[etype].data['link_prediction_score'])
            



# MLP predictor for link classification
class Classify_MLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, n_classes)
        self.activation = nn.Sequential(nn.Sigmoid())

    def apply_edges(self, edges):
        x = torch.cat([edges.src['classify_h'], edges.dst['classify_h']], 1)
        y = self.activation(self.W(x))
        return {'classify_score': y}

    def forward(self, graph, h):
        with graph.local_scope():

            graph.ndata['classify_h'] = h   # 一次性为所有节点类型的'h'赋值
            graph.apply_edges(self.apply_edges)
            return graph.edata['classify_score']




# MLP predictor for link prediction
class LinkPrediction_MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)
        self.activation = nn.Sequential(nn.ReLU())
        


    def apply_edges(self, edges):
        h_u = edges.src['linkprediciton_h']
        h_v = edges.dst['linkprediciton_h']
        score = self.activation(self.W(torch.cat([h_u, h_v], 1)))
        return {'linkprediction_score': score}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['linkprediciton_h'] = h   
            # 只预测指定etype的边
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['linkprediction_score']



class Linear_Classifier(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2))
        
        '''self.W1_TF = nn.Linear(128, 32)
        self.W2_TF = nn.Linear(128, 32)

        self.W3_TF = nn.Linear(64, 16)
        self.W4_TF = nn.Linear(64, 16)
        #self.W5_TF = nn.Linear(8, 32)

        self.W1_tg = nn.Linear(128, 32)
        self.W2_tg = nn.Linear(128, 32)

        self.W3_tg = nn.Linear(64, 16)
        self.W4_tg = nn.Linear(64, 16)
        #self.W5_tg = nn.Linear(8, 16)'''

        self.TF_w1 = nn.Linear(256,128)
        self.TF_w2 = nn.Linear(128,64)
        self.TF_w3 = nn.Linear(64,32)

        self.tg_w1 = nn.Linear(256,128)
        self.tg_w2 = nn.Linear(128,64)
        self.tg_w3 = nn.Linear(64,32)
        
        self.classifier = Classify_MLPPredictor(32, 2)

    def forward(self, graph, g_x):

        h_TF = g_x['TF']
        h_tg = g_x['tg']

        h_TF_1 = self.act1(self.TF_w1(h_TF))
        h_tg_1 = self.act1(self.tg_w1(h_tg))
        
        h_TF_2 = self.act1(self.TF_w2(h_TF_1))
        h_tg_2 = self.act1(self.tg_w2(h_tg_1))
        
        h_TF_3 = self.act1(self.TF_w3(h_TF_2))
        h_tg_3 = self.act1(self.tg_w3(h_tg_2))

        #h_TF = torch.cat((h_TF_1, h_TF_2),1)
        #h_tg = torch.cat((h_tg_1, h_tg_2),1)


        h = {'TF':h_TF_3, 'tg':h_tg_3}
            
        dec_g = graph['TF', : ,'tg']
        edge_label = dec_g.edata[dgl.ETYPE]
        
        edge_label = edge_label.to(torch.int64)
        score = self.classifier(dec_g, h)

        return score
    

class CNN_Classifier(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2))
        self.classifier = Classify_MLPPredictor(253, 2)
        # padding set to 1 and 2 can make sure the output feature size is same to the input feature size
        self.CNNlayer_1_TF = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 32, bias = True)
        self.CNNlayer_1_tg = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 32, bias = True)
        self.CNNlayer_2_TF = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 16, bias = True)
        self.CNNlayer_2_tg = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 16, bias = True)
        self.CNNlayer_3_TF = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 8,  bias = True)
        self.CNNlayer_3_tg = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 8,  bias = True)
        self.CNNlayer_4_TF = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 4,  bias = True)
        self.CNNlayer_4_tg = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 4,  bias = True)

    def forward(self, graph, g_x):
        
        # the number of edges filtered from link prediction model is not fixed
        # so the number of TF/tg may be change 
        TF_num, TF_feature_size = g_x['TF'].shape
        tg_num, tg_feature_size = g_x['tg'].shape

        h_TF_o = torch.reshape(g_x['TF'], (TF_num,1,TF_feature_size))
        h_tg_o = torch.reshape(g_x['tg'], (tg_num,1,tg_feature_size))
    
        h_TF = self.act1(self.CNNlayer_1_TF(h_TF_o))
        h_tg = self.act1(self.CNNlayer_1_tg(h_tg_o))
        
        h_TF = self.act1(self.CNNlayer_2_TF(h_TF_o))
        h_tg = self.act1(self.CNNlayer_2_tg(h_tg_o))

        h_TF = self.act1(self.CNNlayer_3_TF(h_TF_o))
        h_tg = self.act1(self.CNNlayer_3_tg(h_tg_o))

        h_TF = self.act1(self.CNNlayer_4_TF(h_TF_o))
        h_tg = self.act1(self.CNNlayer_4_tg(h_tg_o))

        TF_num, a, TF_feature_size = h_TF.shape
        tg_num, a, tg_feature_size = h_tg.shape

        
        h_TF = torch.reshape(h_TF, (TF_num, TF_feature_size))
        h_tg = torch.reshape(h_tg, (tg_num, tg_feature_size))

        h = {'TF':h_TF, 'tg':h_tg}
        
        dec_g = graph['TF', : ,'tg']
        edge_label = dec_g.edata[dgl.ETYPE]
        
        edge_label = edge_label.to(torch.int64)
        score = self.classifier(dec_g, h)

        return score


class HeteroLinear_v2(nn.Module):
    def __init__(self, in_feats:list, out_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_feats[0], out_dim)
        self.linear_2 = nn.Linear(in_feats[1], out_dim)
        self.linear_3 = nn.Linear(in_feats[2], out_dim)
        self.linear_4 = nn.Linear(in_feats[3], out_dim)
        

    def forward(self, inputs:dict):
        h_name = []
        h_list = []
        h = {}
        for k,v in inputs.items():
            h_name.append(k)
            h_list.append(v)

        h[h_name[0]] = self.linear_1(h_list[0])
        h[h_name[1]] = self.linear_2(h_list[1])
        h[h_name[2]] = self.linear_3(h_list[2])
        h[h_name[3]] = self.linear_4(h_list[3])

        return h




'''class SVM_Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.clf = svm.SVC()'''

'''  
class LSTM_Classifier(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2))
        self.classifier = Classify_MLPPredictor(16, 2)
        self.LSTMlayer = nn.

        def forward(self, graph, g_x):
            print(graph)

            h_TF = self.CNNlayer(g_x['TF'])
            h_tg = self.CNNlayer(g_x['tg'])

            h = {'TF':h_TF, 'tg':h_tg}
            dec_g = graph['TF', : ,'tg']
            edge_label = dec_g.edata[dgl.ETYPE]
            
            edge_label = edge_label.to(torch.int64)
            score = self.classifier(dec_g, h)

            return score'''


# 定义layer时，要用self.layer，否则即使添加进self.gcn_list，定义的模型也没有参数层
class HeteroGCN(nn.Module):
    def __init__(self, etype, gcn_name, in_feats, out_feats):
        super().__init__()
        self.gcn_list = []
        self.gcn_num_dict = {}
        self.gcn_name = gcn_name
        idx = 0
        
        for k,v in etype.items():
            self.gcn_num_dict[k] = idx
            idx += 1 
            if gcn_name == 'TAG':
                self.layer = dglnn.TAGConv(in_feats, out_feats)
                self.gcn_list.append(self.layer)
            elif gcn_name == 'GIN':
                self.lin = nn.Linear(in_feats, out_feats)
                self.layer = dglnn.GINConv(self.lin, 'max')
                self.gcn_list.append(self.layer)
            elif self.gcn_name == 'SAGE':
                self.layer = dglnn.SAGEConv((in_feats, in_feats), out_feats, 'mean')
                self.gcn_list.append(self.layer)
            elif self.gcn_name == 'GAT':
                self.layer = dglnn.SAGEConv(in_feats, out_feats)
                self.gcn_list.append(self.layer)
            elif self.gcn_name == 'GraphConv':
                self.layer = dglnn.GraphConv(in_feats, out_feats)
                self.gcn_list.append(self.layer)
        #print(len(self.gcn_list), 'len')    


    def forward(self, graph, inputs):
        etypes = graph.canonical_etypes
        h_node = {}
        for i in range(len(etypes)):
            u,e,d = etypes[i]
            h_u = inputs[u]
            h_d = inputs[d]
            
            etype_subgraph = graph.edge_type_subgraph([e])
            
            if u == d:
                h_input = h_u
            else:
                h_input = {u:h_u, d:h_d}
            
            idx = self.gcn_num_dict[e]
                
            if self.gcn_name == 'TAG':
                # graph have to be a homogeneous graph , X 

                h_d_out = self.gcn_list[idx](etype_subgraph, h_input)
            elif self.gcn_name == 'GIN':
                h_d_out = self.gcn_list[idx](etype_subgraph, h_input)
            elif self.gcn_name == 'SAGE':
                if u == d:
                    h_d_out = self.gcn_list[idx](etype_subgraph, (h_input, h_input))
                else:
                    h_d_out = self.gcn_list[idx](etype_subgraph, (h_input[u], h_input[d]))
            elif self.gcn_name == 'GraphConv':
                h_d_out = self.gcn_list[idx](etype_subgraph, h_input)
                                
            if d in h_node.keys():
                h_node[d] = h_node[d] + h_d_out
            else:
                h_node[d] = h_d_out
        return h_node         
        
