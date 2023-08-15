import torch
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from model.GraphLinkPredictionMoudle import GCNStructureParsing
from model.Dexined import DexiNed
from model.SHG import StackedHourglassNetwork
from util.geocal import (
    getDexinedFusedEdgeMap,
    getpidinetEdgeMap,
    SampleGraphVertex,
    GraphEdgeSample,
    SampleNodesFeature,
)
from model.pidinet.pidinet import PiDiNet
from model.pidinet.config import config_model


class StructureParsingNet(torch.nn.Module):
    """Definition of the StructureParsingNet network."""

    def __init__(self, cfg):
        super(StructureParsingNet, self).__init__()

        self.gnn_in_channels = cfg.gnn_in_channels
        self.pe_length = cfg.pe_length
        self.heatmap_size = cfg.heatmap_size
        self.train_batch_size = cfg.train_batch_size
        self.max_junction_num = cfg.max_junction_num
        self.junc_score_thresh = cfg.junc_score_thresh
        self.train_flag = cfg.train_flag
        self.EdgeBackboneName = cfg.EdgeBackboneName
        self.avg_edge_thresh = cfg.avg_edge_thresh
        self.max_line_length = cfg.max_line_length
        self.continue_edge_thresh = cfg.continue_edge_thresh
        self.angle_thresh = cfg.angle_thresh
        self.Edge_Sample_method = cfg.Edge_Sample_method
        self.Max_range = cfg.Max_Range
        self.Max_Neiborghoods = cfg.Max_Neiborghoods

        if self.EdgeBackboneName == "dexined":
            self.ShapeBackbone = DexiNed(cfg)


        # elif self.EdgeBackboneName == "pidinet":
        #     pdcs = config_model(cfg.pidinet_pdcs)
        #     self.ShapeBackbone = PiDiNet(
        #         cfg.pidinet_inplane, pdcs=pdcs, dil=cfg.pidinet_dil, sa=cfg.pidinet_sa
        #     )
            # conv_weights, bn_weights, relu_weights = self.ShapeBackbone.get_weights()
            # self.p_enc_2d_model_sum = Summer(PositionalEncoding2D(cfg.SHGoutchannels))

        # self.SemanticBackbone = StackedHourglassNetwork(
        #     depth=cfg.SHG_depth,
        #     num_stacks=cfg.SHG_num_stacks,
        #     num_blocks=cfg.SHG_num_blocks,
        #     num_feats=cfg.SHG_output_layers,
        #     JuncPred_output_channels_list = cfg.JuncPred_shg_channels_list
        # )
        # self.p_enc_2d_model_sum = Summer(
        #         PositionalEncoding2D(cfg.SHGoutchannels)
        #     )
        self.StructureParsingGCN = GCNStructureParsing(
            cfg.gnn_in_channels, cfg.gnn_out_channels, cfg.EncoderName
        )

    def forward(self, images):
        Bsize = images.shape[0]

        ResultEdgeMaps, ResultJuncMaps, FeatureShape = self.ShapeBackbone(images)
        # 8 24 512 512 8 24 256 256 8 24 128 128 8 24 64 64
        # ResultJuncMaps, FeatureSemantic = self.SemanticBackbone(images)
        # 256 128 128

        if self.train_flag == "junc":
            return ResultEdgeMaps, ResultJuncMaps

        else:
            with torch.no_grad():
                junc_preds, junc_scores = SampleGraphVertex(
                    ResultJuncMaps, self.max_junction_num, self.junc_score_thresh
                )

                GraphVertex_xy = [
                    junc_preds[index_i].detach().cpu().numpy()
                    for index_i in range(Bsize)
                ]

                if self.EdgeBackboneName == "pidinet":
                    # Edge_Maps = getpidinetEdgeMap(ResultEdgeMaps, Bsize)
                    Edge_Maps = torch.squeeze(ResultEdgeMaps[-1]).detach().cpu().numpy()
                elif self.EdgeBackboneName == "dexined":
                    Edge_Maps = getDexinedFusedEdgeMap(ResultEdgeMaps, Bsize)

            # return Edge_Maps

                graph_edge = GraphEdgeSample(
                    GraphVertex_xy,
                    Edge_Maps,
                    self.max_line_length,
                    self.avg_edge_thresh,
                    self.continue_edge_thresh,
                    self.angle_thresh,
                    Bsize,
                    samplemethod=self.Edge_Sample_method,
                    Max_Range=self.Max_range,
                    Max_Neiborghoods=self.Max_Neiborghoods,
                )

                graph_edge = [
                    torch.tensor(graph_edge[index_i]).permute(1, 0).type(torch.long)
                    for index_i in range(Bsize)
                ]

                query_edge_index = [
                    graph_edge[index_i].permute(1, 0).detach().cpu().numpy()
                    for index_i in range(Bsize)
                ]

            # FeatureSemantic = self.p_enc_2d_model_sum(FeatureSemantic.permute(
            #     0, 2, 3, 1)).permute(0, 3, 1, 2)    
            NodeFeatures = SampleNodesFeature(
                GraphVertex_xy, FeatureShape, Bsize, images.device
            )
            link_preds = [
                self.StructureParsingGCN(
                    NodeFeatures[index_i],
                    graph_edge[index_i].to(images.device),
                    query_edge_index[index_i],
                )
                for index_i in range(Bsize)
            ]

            return GraphVertex_xy, graph_edge, query_edge_index, link_preds
