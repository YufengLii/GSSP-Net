import numpy as np
import torch
from torch.nn import (Linear, ReLU, 
                      Sequential, BatchNorm1d as BN)
import torch.nn.functional as F
from torch_geometric.nn import ( SAGEConv, GINConv, Linear, BatchNorm, TransformerConv, EdgeConv, TopKPooling, EdgePooling)
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import add_self_loops
# from torch_geometric.nn import TopKPooling
from torch_geometric.nn.models import EdgeCNN

class myEdgeCNN(torch.nn.Module):
    def __init__(self, INchannels, Outchannels):
        super().__init__()

        self.model1 = EdgeCNN(in_channels=INchannels, hidden_channels=1024, num_layers =3, out_channels =Outchannels*2, dropout =0.1 )

    def forward(self, x, edge_index):
        x = self.model1(x, edge_index)
        return x

class GCN_Encoder(torch.nn.Module):
    def __init__(self, INchannels, Outchannels):
        super().__init__()
        self.lin = Linear(INchannels, Outchannels*2, bias=True)
        self.conv1 = TransformerConv((-1, -1), Outchannels*4, heads = 1, bias =True)
        self.bn1 = BatchNorm(Outchannels*4)
        self.conv2 = TransformerConv((-1, -1), Outchannels*2, heads = 1, bias =True)
        self.bn2 = BatchNorm(Outchannels*2)
        self.conv3 = TransformerConv((-1, -1), Outchannels*2, heads = 1, bias =True)
        self.bn3 = BatchNorm(Outchannels*2)
    
    def forward(self, x, edge_index):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.lin(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu() # better
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = x.relu()
        return x  # n 704

class GIN_Encoder(torch.nn.Module):
    def __init__(self, INchannels, Outchannels):  
        super().__init__()
        self.lin = Linear(INchannels, Outchannels*2, bias=True)
        
        self.conv1 = GINConv( nn=Sequential(
                Linear(Outchannels*2, Outchannels*4),
                ReLU(),
                Linear(Outchannels*4, Outchannels*4),
                ReLU(),
                BN(Outchannels*4)), eps=0, train_eps=True )
        
        self.bn1 = BatchNorm(Outchannels*4)
        self.conv2 = GINConv( nn=Sequential(
                Linear(Outchannels*4, Outchannels*4),
                ReLU(),
                Linear(Outchannels*4, Outchannels*4),
                ReLU(),
                BN(Outchannels*4)), eps=0, train_eps=True )
        self.bn2 = BatchNorm(Outchannels*4)    
        
        self.conv3 = GINConv( nn=Sequential(
                Linear(Outchannels*4, Outchannels*2),
                ReLU(),
                Linear(Outchannels*2, Outchannels*2),
                ReLU(),
                BN(Outchannels*2)) , eps=0, train_eps=True)
        self.bn3 = BatchNorm(Outchannels*2)  
        
    def forward(self, x, edge_index):    
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu() # better
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)

        return x

class SAGE_Encoder(torch.nn.Module):
    
    def __init__(self, INchannels, Outchannels):  
        super().__init__()
        self.lin = Linear(INchannels, Outchannels*2, bias=True)
        self.conv1 = SAGEConv((-1, -1), Outchannels*4, aggr='max')
        self.bn1 = BatchNorm(Outchannels*4)
        self.conv2 = SAGEConv((-1, -1), Outchannels*4, aggr='max')
        self.bn2 = BatchNorm(Outchannels*4)        
        self.conv3 = SAGEConv((-1, -1), Outchannels*2, aggr='max')
        self.bn3 = BatchNorm(Outchannels*2)     
        # self.conv4 = SAGEConv((-1, -1), INchannels*2, aggr='max')
        # self.bn4 = BatchNorm(INchannels*2)         
           
    def forward(self, x, edge_index):    
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu() # better
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # x = self.bn3(x)
        # x = x.relu()
        # x = self.conv4(x, edge_index)
        # x = self.bn4(x)
        return x

class DGCNN_Encoder(torch.nn.Module):
    def __init__(self, INchannels, Outchannels):  
        super().__init__()
        self.lin = Linear(INchannels, Outchannels, bias=True)
        
        # self.pool1 = EdgePooling(1024)
        # self.pool2 = EdgePooling(1024)
        # self.pool3 = EdgePooling(1024, ratio=0.8)
        # x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        self.conv1 = EdgeConv( Sequential(
                Linear(Outchannels*2, Outchannels*4),
                ReLU(),
                Linear(Outchannels*4, Outchannels*2),
                ReLU(),
                BN(Outchannels*2)), k = 16)
        self.bn1 = BatchNorm(Outchannels*2)
        
        self.conv2 = EdgeConv( Sequential(
                Linear(Outchannels*4, Outchannels*4),
                ReLU(),
                Linear(Outchannels*4, Outchannels*2),
                ReLU(),
                BN(Outchannels*2)), k = 8 )
        self.bn2 = BatchNorm(Outchannels*2)    
        
        self.conv3 = EdgeConv( Sequential(
                Linear(Outchannels*4, Outchannels*2),
                ReLU(),
                Linear(Outchannels*2, Outchannels*2),
                ReLU(),
                BN(Outchannels*2)) , k = 8)
        self.bn3 = BatchNorm(Outchannels*2)  

    def forward(self, x, edge_index):    
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu() # better
        # x, edge_index, _, _ = self.pool1(x, edge_index, torch.zeros(x.shape[0], dtype=torch.int) )

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        # x, edge_index,_,_,_,_ = self.pool2(x, edge_index, None, None)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        return x

class GCNStructureParsing(torch.nn.Module):
    def __init__(self, Inchannels, Outchannels, EncoderName):
        super().__init__()
        
        if EncoderName == "GIN":
            self.encoder = GIN_Encoder(Inchannels, Outchannels)
        elif EncoderName == "Transformer":
            self.encoder = GCN_Encoder(Inchannels, Outchannels)
        elif EncoderName == "DGCNN":
            self.encoder = DGCNN_Encoder(Inchannels, Outchannels)
        elif EncoderName == "SAGE":
            self.encoder = SAGE_Encoder(Inchannels, Outchannels)
        elif EncoderName == "myEdgeCNN":
            self.encoder = myEdgeCNN(Inchannels, Outchannels)
        
        self.decoder = GNNDecoder(Outchannels*2)

    def forward(self, x, edge_index, junc_index_pair):
        
        x = self.encoder(x, edge_index)
        x = self.decoder(x, junc_index_pair)
        return torch.sigmoid(x)

class GNNDecoder(torch.nn.Module):

    def __init__(self, dim_in):
        super().__init__()

        self.layer_post_mp = MLP(
                new_layer_config(dim_in * 2, 1, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
        self.decode_module_cat = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=1))
        
    def forward(self, x, junc_index_pair):

        node_start = x[junc_index_pair[:,0]] # n x 704
        node_end = x[junc_index_pair[:,1]]
        pred = self.decode_module_cat(node_start, node_end)
        return pred.squeeze()




