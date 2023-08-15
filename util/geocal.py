import torch
import numpy as np
import cv2
from numba import jit
import torch.nn.functional as F
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
from PIL import Image
import os
from util.edge_eval_python.impl.toolbox import conv_tri, grad2
from ctypes import *
from scipy import ndimage
# import warnings
# warnings.simplefilter('ignore', category=RuntimeWarning) 

def NonMaxSup(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS

def myedgenms(image):

    dx = np.array([[-1,0,1],
                [-1,0,1],
                [-1,0,1]])
    dy = np.array([[-1,-1,-1],
                [0,0,0],
                [1,1,1]])
    dx_img = ndimage.convolve(image, dx)
    dy_img = ndimage.convolve(image, dy)
    gradient = np.degrees(np.arctan2(dy_img,dx_img))

    img_nms = NonMaxSup(image, gradient)

    return img_nms

solver = cdll.LoadLibrary("util/edge_eval_python/cxx/lib/solve_csa.so")
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]

def nms_process_one_image(image, save_path=None, save=False):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    edge = np.float32(edge)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    # np.seterr(divide="ignore", invalid="ignore")

    oxx_preprocess = oxx + 1e-5
    oxx_preprocess[oxx_preprocess==0] = 1e-5
    # aaa = oyy * np.sign(-oxy)

                    
    # try :
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / oxx_preprocess), np.pi)
    # except:
    #     oxx_preprocess = oxx_preprocess + 1e-5
    #     ori = np.mod(np.arctan(oyy * np.sign(-oxy) / oxx_preprocess), np.pi)
    #     print('-----')

    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    # r, s, m, w, h = 1, 3, float(1.01), int(out.shape[1]), int(out.shape[0])

    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    # edge = np.round(out * 255).astype(np.uint8)

    return out

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):

    img = np.float32(img)
    img = (img - np.min(img)) * (img_max - img_min) / (
        (np.max(img) - np.min(img)) + epsilon
    ) + img_min
    return img

def getpidinetEdgeMap(ResultEdgeMaps, b_size):
    
    out = np.zeros((b_size, 512, 512), dtype=np.float32)
    
    # out = torch.squeeze(ResultEdgeMaps[-1]).detach().cpu().numpy()
    
    
    for idx in range(b_size):
        tmp = ResultEdgeMaps[-1][idx,0,:,:].detach().cpu().numpy()
        out[idx, :, :] = tmp

        result = Image.fromarray((tmp * 255).astype(np.uint8))
        result.save(str(idx)+".png")
        
    return out

def getDexinedFusedEdgeMap(ResultEdgeMaps, b_size):

    edge_maps = []
    for i in ResultEdgeMaps:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    ResultEdgeMaps = np.array(edge_maps)
    fuse_out = np.zeros((b_size, 512, 512), dtype=np.float32)

    for idx in range(b_size):
        tmp = ResultEdgeMaps[:, idx, ...]
        tmp = np.squeeze(tmp)
        for i in range(tmp.shape[0]):
            if i == 6:
                tmp_img = tmp[i]
                tmp_img = np.uint8(image_normalization(tmp_img))
                tmp_img = cv2.bitwise_not(tmp_img)
                fuse = (255.0 - tmp_img) / 255.0
                # fuse = tmp_img / 255.0
                # fuse = myedgenms(fuse)

                try:
                    fuse = nms_process_one_image(fuse)
                except:
                    print("Exception use raw edge")
                
                # edge = np.round(fuse * 255).astype(np.uint8)
                # cv2.imwrite("nms.png", edge)
                # # fuse = tmp_img / 255.0
        fuse_out[idx, :, :] = fuse
    return fuse_out

def SampleGraphVertex(ResultJuncMaps, max_junc_num, thresh_junc):

    junc_preds = []
    junc_scores = []

    b = ResultJuncMaps["jmap"].shape[0]
    for i in range(b):

        jmap = ResultJuncMaps["jmap"][i]
        joff = ResultJuncMaps["joff"][i]
        junc_pred, junc_score = calc_junction(
            jmap, joff, thresh=thresh_junc, top_K=max_junc_num
        )  # self.junc_max_num

        junc_preds.append(junc_pred)
        junc_scores.append(junc_score)

    return junc_preds, junc_scores

def calc_junction(jmap, joff, thresh, top_K):
    h, w = jmap.shape[-2], jmap.shape[-1]
    score = jmap.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // w, indices % w

    junc = torch.cat((x[:, None], y[:, None]), dim=1) + joff[indices] + 0.5

    junc[:, 0] = junc[:, 0].clamp(min=0, max=w - 1e-4)
    junc[:, 1] = junc[:, 1].clamp(min=0, max=h - 1e-4)

    return junc, score

def my_collate_fn(data):

    unit_image = []
    unit_edge = []
    unit_jmap = []
    unit_joff = []
    file_path = []
    for data_bi in data:

        unit_image.append(torch.tensor(data_bi[0]))
        unit_edge.append(torch.tensor(data_bi[1]))
        unit_jmap.append(torch.tensor(data_bi[2]))
        unit_joff.append(torch.tensor(data_bi[3]))
        file_path.append(torch.tensor(data_bi[4]))
        
    return unit_image, unit_edge, unit_jmap, unit_joff, file_path

@jit(nopython=True)
def lineiterator(start, end):
    lenth_line = np.sqrt(
        np.power(start[0] - end[0], 2) + np.power(start[1] - end[1], 2)
    )
    sample_points = []
    sample_points.append((int(start[0]), int(start[1])))
    if lenth_line <= 2:
        sample_points.append((int(end[0]), int(end[1])))
        return sample_points
    if lenth_line <= 10:
        for i in range(1, int(lenth_line)):
            yi = (end[0] - start[0]) * i / lenth_line + start[0]
            xi = (end[1] - start[1]) * i / lenth_line + start[1]
            sample_points.append((int(yi), int(xi)))
        sample_points.append((int(end[0]), int(end[1])))
        return sample_points
    if lenth_line > 10:
        for i in range(1, 10):
            yi = (end[0] - start[0]) * i / 10 + start[0]
            xi = (end[1] - start[1]) * i / 10 + start[1]
            sample_points.append((int(yi), int(xi)))
        sample_points.append((int(end[0]), int(end[1])))
        return sample_points


    # return sample_points

@jit(nopython=True)
def angleAB(a, b):
    """return rotation angle from vector a to vector b, in degrees.
    Args:
        a : np.array vector. format (x,y)
        b : np.array vector. format (x,y)
    Returns:
        angle [float]: degrees. 0~180
    """

    unit_vector_1 = a / np.sqrt(np.power(a[0], 2) + np.power(a[1], 2))
    unit_vector_2 = b / np.sqrt(np.power(b[0], 2) + np.power(b[1], 2))

    dot_product = (
        unit_vector_1[0] * unit_vector_2[0] + unit_vector_1[1] * unit_vector_2[1]
    )

    angle = np.arccos(dot_product)
    angle = angle / np.pi * 180

    return angle

@jit(nopython=True)
def GraphEdgeSampler_SturtureGuided(
    nodes,
    lmap,
    R_thresh=256.0 + 128,
    avg_edge_thresh=0.75,
    continue_edge_thresh=0.3,
    angle_thresh=6.0,
):

    node_num = nodes.shape[0]
    graph_edge = []

    for j in range(node_num):

        adj_j = []
        adj_j_out = []
        adj_j_score = []
        dydx = []
        center = nodes[j, :]
        O_xy = nodes - center
        R_ = np.zeros(O_xy.shape[0], dtype=float)
        for k in range(R_.shape[0]):
            R_[k] = np.linalg.norm(O_xy[k], ord=2)
        R_index_sort = np.argsort(R_)

        # 最近的N个点， 满足条件则添加到邻域
        for i in range(1, R_index_sort.shape[0]):

            if R_[R_index_sort[i]] < R_thresh:

                check_index = R_index_sort[i]
                start_int = (nodes[check_index, 1], nodes[check_index, 0])
                end_int = (center[1], center[0])
                discrete_line = lineiterator(start_int, end_int)

                scores = []
                for yixi in discrete_line:
                    if (
                        yixi[0] - 1 >= 0
                        and yixi[0] + 1 <= 511
                        and yixi[1] - 1 >= 0
                        and yixi[1] + 1 <= 511
                    ):
                        neibor_array = lmap[
                            yixi[0] - 1 : yixi[0] + 1, yixi[1] - 1 : yixi[1] + 1
                        ]
                        scores.append(neibor_array.max())
                    else:
                        scores.append(lmap[yixi])

                # scores = [lmap[yixi] for yixi in discrete_line]

                discard_flag = False
                if len(scores) > 5:
                    scores_TF = np.array(scores) - continue_edge_thresh
                    num_false = 0
                    for jjj in range(len(scores)):
                        if scores_TF[jjj] < 0:
                            num_false += 1
                        else:
                            num_false = 0
                        if num_false >= 5:
                            num_false = 0
                            discard_flag = True
                            break

                scores_avg = sum(scores) / len(scores)
                if scores_avg > avg_edge_thresh and discard_flag is False:

                    adj_j.append([j, R_index_sort[i]])
                    adj_j_score.append(scores_avg)
                    dydx.append(
                        [
                            nodes[check_index, 0] - center[0],
                            nodes[check_index, 1] - center[1],
                        ]
                    )

        if len(adj_j_score) >= 1:
            adj_index_sort = np.argsort(np.array(adj_j_score))

            for index in range(len(adj_j_score) - 1, -1, -1):
                if len(adj_j_out) <= 3:
                    adj_j_out.append(adj_j[adj_index_sort[index]])
                else:
                    check_index = adj_j[adj_index_sort[index]][1]
                    check_point = np.array(
                        [nodes[check_index, 0], nodes[check_index, 1]]
                    )
                    adj_points = np.zeros((len(adj_j_out), 2), dtype=float)
                    for adj_i in range(len(adj_j_out)):
                        adj_points[adj_i, 0] = nodes[adj_j_out[adj_i][1], 0]
                        adj_points[adj_i, 1] = nodes[adj_j_out[adj_i][1], 1]

                    angle_differ = np.zeros(adj_points.shape[0], dtype=float)
                    for x in range(adj_points.shape[0]):
                        angle_differ[x] = angleAB(
                            check_point - center, adj_points[x, :] - center
                        )

                    angle_index_sort = np.argsort(angle_differ)

                    if angle_differ[angle_index_sort[3]] > angle_thresh:
                        # if angle_differ[angle_index_sort[3]] > angle_thresh and len(adj_j_out)<4:
                        adj_j_out.append(adj_j[adj_index_sort[index]])

        else:
            adj_j_out = adj_j

        graph_edge.extend(adj_j_out)

    # undirected graph
    graph_edge_out = []
    for edge_ii in graph_edge:
        edge_ii_a = [edge_ii[0], edge_ii[1]]
        edge_ii_b = [edge_ii[1], edge_ii[0]]

        if edge_ii_a not in graph_edge_out and edge_ii_b not in graph_edge_out:
            graph_edge_out.append(edge_ii_a)
    return graph_edge_out

@jit(nopython=True)
def GraphEdgeSampler_K_Neiborghoods(nodes, Max_Neiborghoods=8):

    node_num = nodes.shape[0]
    graph_edge = []

    for j in range(node_num):

        adj_j = []
        adj_j_out = []
        center = nodes[j, :]
        O_xy = nodes - center
        R_ = np.zeros(O_xy.shape[0], dtype=float)
        for k in range(R_.shape[0]):
            R_[k] = np.linalg.norm(O_xy[k], ord=2)
        R_index_sort = np.argsort(R_)

        for i in range(1, R_index_sort.shape[0]):

            adj_j.append([j, R_index_sort[i]])

            if len(adj_j) >= Max_Neiborghoods:
                break

        graph_edge.extend(adj_j)

    # undirected graph
    graph_edge_out = []
    for edge_ii in graph_edge:
        edge_ii_a = [edge_ii[0], edge_ii[1]]
        edge_ii_b = [edge_ii[1], edge_ii[0]]

        if edge_ii_a not in graph_edge_out and edge_ii_b not in graph_edge_out:
            graph_edge_out.append(edge_ii_a)

    return graph_edge_out

@jit(nopython=True)
def GraphEdgeSampler_R_Range(nodes, Max_Range=64):

    node_num = nodes.shape[0]
    graph_edge = []

    for j in range(node_num):

        adj_j = []
        adj_j_out = []
        center = nodes[j, :]
        O_xy = nodes - center
        R_ = np.zeros(O_xy.shape[0], dtype=float)
        for k in range(R_.shape[0]):
            R_[k] = np.linalg.norm(O_xy[k], ord=2)
        R_index_sort = np.argsort(R_)

        for i in range(1, R_index_sort.shape[0]):

            adj_j.append([j, R_index_sort[i]])

            if len(adj_j) >= Max_Range:
                break

        graph_edge.extend(adj_j)

    # undirected graph
    graph_edge_out = []
    for edge_ii in graph_edge:
        edge_ii_a = [edge_ii[0], edge_ii[1]]
        edge_ii_b = [edge_ii[1], edge_ii[0]]

        if edge_ii_a not in graph_edge_out and edge_ii_b not in graph_edge_out:
            graph_edge_out.append(edge_ii_a)

    return graph_edge_out

def GraphEdgeSample(
    vertexxy,
    edgemaps,
    max_line_length,
    avg_edge_thresh,
    continue_edge_thresh,
    angle_thresh,
    Bsize,
    samplemethod="ours",
    Max_Range=64,
    Max_Neiborghoods=8,
):
    if samplemethod == "ours":
        graph_edge = [
            GraphEdgeSampler_SturtureGuided(
                vertexxy[index_i] * 4.0,
                edgemaps[index_i, :, :],
                max_line_length,
                avg_edge_thresh,
                continue_edge_thresh,
                angle_thresh,
            )
            for index_i in range(Bsize)
        ]
    elif samplemethod == "Kneiborghood":

        graph_edge = [
            GraphEdgeSampler_K_Neiborghoods(
                vertexxy[index_i] * 4.0,
                Max_Neiborghoods,
            )
            for index_i in range(Bsize)
        ]

    elif samplemethod == "Rrange":

        graph_edge = [
            GraphEdgeSampler_R_Range(
                vertexxy[index_i] * 4.0,
                Max_Range,
            )
            for index_i in range(Bsize)
        ]

    return graph_edge

def SampleNodesFeature(GraphVertex_xy, FeatureShape, Bsize, device):

    # 8 24 512 512 8 24 256 256 8 24 128 128 8 24 64 64
    # 256 128 128
    # 96 + 256
    sample_biyx = [
        torch.tensor((GraphVertex_xy[index_i][:, [1, 0]] - 64.0) / 64.0)
        .type(torch.float32)
        .to(device)
        for index_i in range(Bsize)
    ]
    
    ### Multi Level Feature
    Nodefeature1 = [
        F.grid_sample(
            FeatureShape[0][index_i].unsqueeze(0),
            sample_biyx[index_i].unsqueeze(0).unsqueeze(0),
            align_corners=True,
        )
        .squeeze()
        .permute(1, 0)
        for index_i in range(Bsize)
    ]

    Nodefeature2 = [
        F.grid_sample(
            FeatureShape[1][index_i].unsqueeze(0),
            sample_biyx[index_i].unsqueeze(0).unsqueeze(0),
            align_corners=True,
        )
        .squeeze()
        .permute(1, 0)
        for index_i in range(Bsize)
    ]

    # Nodefeature3 = [
    #     F.grid_sample(
    #         FeatureShape[2][index_i].unsqueeze(0),
    #         sample_biyx[index_i].unsqueeze(0).unsqueeze(0),
    #         align_corners=True,
    #     )
    #     .squeeze()
    #     .permute(1, 0)
    #     for index_i in range(Bsize)
    # ]

    Nodefeature4 = [
        F.grid_sample(
            FeatureShape[3][index_i].unsqueeze(0),
            sample_biyx[index_i].unsqueeze(0).unsqueeze(0),
            align_corners=True,
        )
        .squeeze()
        .permute(1, 0)
        for index_i in range(Bsize)
    ]
    
    # Nodefeature5 = [
    #     F.grid_sample(
    #         FeatureSemantic[index_i].unsqueeze(0),
    #         sample_biyx[index_i].unsqueeze(0).unsqueeze(0),
    #         align_corners=True,
    #     )
    #     .squeeze()
    #     .permute(1, 0)
    #     for index_i in range(Bsize)
    # ]

    NodeFeatures = [
        torch.cat(
            (
                Nodefeature1[index_i],
                Nodefeature2[index_i],
                # Nodefeature3[index_i],
                Nodefeature4[index_i],
                # Nodefeature5[index_i],
            ),
            1,
        )
        for index_i in range(Bsize)
    ]
    
    return NodeFeatures  # 352

def generateLinkGTbydistance(gt_anno, sample_bi, query_edge_index, step, bid, thresh, withDraw = True):

    link_proposals = [ [sample_bi[edge_index[0]], sample_bi[edge_index[1]] ]  for edge_index in query_edge_index  ]

    gtpos_index = (gt_anno == 1).nonzero()
    gtpos_index = gtpos_index.cpu().numpy()
    gtlink = []

    for aline in link_proposals:
        line = LineString(([aline[0][1], aline[0][0]], [aline[1][1], aline[1][0]]))

        if line.length < 3:
            sample_points = np.array( [[aline[0][1], aline[0][0]], [aline[1][1], aline[1][0]] ] )

        else:
            distance_delta = 2
            distances = np.arange(0, line.length, distance_delta)

            points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
            sample_points = np.array([ ss.xy for ss in points ]).squeeze()

        Y = cdist(sample_points, gtpos_index, 'euclidean').min(1)
        
        if Y.max() >= 2.0 or Y.mean() >=2.0:
            gtlink.append(0)
        else:
            gtlink.append(1)


    if withDraw:

        img = np.zeros([512,512,3],dtype=np.uint8)

        for i in range(len(gtlink)):
        
           start_point = ( int(link_proposals[i][0][0]), int(link_proposals[i][0][1]) )
           end_point = ( int(link_proposals[i][1][0]), int(link_proposals[i][1][1]) )
        
           if gtlink[i] == 1:
        
               img = cv2.line(img, start_point, end_point, (0,0,255), 1)
        
        cv2.imwrite("./log/gtlink/" + str(step) + '-' + str(bid) + '-' + str(int(thresh*100)) +'.png' , img)

    return gtlink




def checkRenderEdge(query_edge_index, link_preds, device):

    render_edge_batch = []
    render_score_batch = [] 

    for bi in range(len(link_preds)):
        query_edge_index_bi = query_edge_index[bi]  # numpy (n, 2)
        junc_index_source = query_edge_index_bi[:,0]

        split_index = []
        for ji in range(junc_index_source.shape[0]-1):
            if junc_index_source[ji] != junc_index_source[ji+1] :

                split_index.append(ji) # [0, index] [index+1, index] [index+1, end]

        link_split_range = []

        for i in range(len(split_index)):

            if i == 0:
                split_range = [0, split_index[i]+1]
                
            else:
                split_range = [split_index[i-1]+1, split_index[i]+1]

            link_split_range.append(split_range)

        split_range = [split_index[-1]+1, junc_index_source.shape[0]]
        link_split_range.append(split_range)
        # print(link_split_range)
        
        check_index_bi = []
        for range_ii in link_split_range:

            if (range_ii[1] - range_ii[0] ) > 2:
                link_score_range = link_preds[bi][range_ii[0]:range_ii[1]]
                # print(link_score_range)
                node_link_sort_index = torch.argsort(link_score_range, descending=True)
                check_index = node_link_sort_index[0:2] + range_ii[0]
                check_index_bi.append(check_index[0])
                check_index_bi.append(check_index[1])
            elif (range_ii[1] - range_ii[0] ) == 2:
                check_index = [range_ii[0], range_ii[1]-1]
                check_index_bi.append(torch.tensor(check_index[0], dtype=torch.int64).to(device)  )
                check_index_bi.append(torch.tensor(check_index[1], dtype=torch.int64).to(device))                        
            else:
                check_index = range_ii[0]
                check_index_bi.append( torch.tensor(check_index, dtype=torch.int64).to(device) )

        check_index_bi = torch.tensor(check_index_bi)
        check_index_bi = check_index_bi.detach().cpu().numpy()

        render_edge_batch.append(query_edge_index[bi][check_index_bi,:])
        render_score_batch.append(link_preds[bi][check_index_bi])

    return  render_edge_batch, render_score_batch