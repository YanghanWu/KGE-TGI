import torch as torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn



class HeteroLinear(nn.Module):
    def __init__(self, in_feats:dict, out_dim) -> None:
        super().__init__()
        self.linear_dict = {}
        for k,v in in_feats.items():
            self.linear_dict[k] = nn.Linear(v, out_dim)
        

    def forward(self, inputs:dict):
        h = {}
        for k, v in inputs.items():
            h[k] = self.linear_dict[k](v)
        
        return h


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

        self.Heterolinear = HeteroLinear({'TF': in_feats, 'tg': in_feats, 'disease': in_feats, 'go': 4338}, size1)

        self.conv1 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size1, size2, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size1, size2, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size1, size2, activation = self.act1)
            },
            aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'regulate' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_1' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_2' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'associate_3' : dglnn.GraphConv(size2, size3, activation = self.act1),
            'go_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1),
            'TF_self_loop': dglnn.GraphConv(size2, size3, activation = self.act1)
            },
            aggregate='sum')
  
        
    def forward(self, graph, inputs):

        # inputs = node features
        # transfer to same size
        h = self.Heterolinear(inputs)
        

        # GraphConv
        h1 = self.conv1(graph, h)
        h_mid = {k: self.act1(v) for k, v in h1.items()}     # 还可以加norm

        h2 = self.conv2(graph, h_mid)

        # h is a dict, key = 'TF', 'disease', 'tg'
        node_types = graph.ntypes
        for i in range(len(node_types)):
            graph.apply_nodes(lambda nodes: {'h': h2[node_types[i]]}, ntype=node_types[i])
       
        return graph




class GraphTGI_classifier(nn.Module):
    def __init__(self, in_features, hidden_features, slope):
        super().__init__()
        self.sage = RGCN_classify(in_features, hidden_features, slope)  
        self.classifier = Classify_MLPPredictor(32, 2)

    def forward(self, g, g_x):
        
        h = self.sage(g, g_x)
   
        # to predict the type of edges, we need to mask the edge type information
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

            graph.ndata['classify_h'] = h   
            graph.apply_edges(self.apply_edges)
            return graph.edata['classify_score']



