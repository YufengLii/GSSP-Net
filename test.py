import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.utils.data as Data
from model.StructureParsing import StructureParsingNet
from config.cfg import parse
from dataset.SSUdataset import DatasetGSSU
from util.rendering import graph_link_render_one_with_threshhold
import cv2
import time
# from util.geocal import drawLinks
from util.geocal import checkRenderEdge


def test(model, loader, cfg, device):

    if cfg.last_epoch != -1:
        checkpoint_file = os.path.join(cfg.checkpointdir,
                                       f'graph-{cfg.last_epoch:03d}.pkl')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    for images, gt_annos, gt_jmap, gt_joff, imagefilename in tqdm(loader):

        imagefileperfix = imagefilename[0].split('.')[0]
        images = images.to(device)

        GraphVertex_xy, graph_edge, query_edge_index, link_preds = model(
            images)

        render_edge_batch, render_score_batch = checkRenderEdge(query_edge_index, link_preds, device)

        for index_i in range(len(link_preds)):

            sample_bi_j = GraphVertex_xy[index_i]
            query_bi_j = render_edge_batch[index_i]  

            render_points = [[
                sample_bi_j[item[1]],
                sample_bi_j[item[0]]
            ] for item in query_bi_j]

            # may need adjust when draw
            render_points = torch.tensor(np.array(render_points) *
                                            4.0).to(device)
            render_alpha = render_score_batch[index_i]

            np.savez(os.path.join("./output/paths/" , imagefileperfix + '.npz' ),  
                    lines = render_points.detach().cpu().numpy() , scores = render_alpha.detach().cpu().numpy()) 

            rendered_result, _ = graph_link_render_one_with_threshhold(render_points,
                                                            render_alpha,
                                                            force_cpu=True,
                                                            canvas_size=512,
                                                            alpha_threshold=0.01,
                                                            colors=None)

            imageout = rendered_result.squeeze().detach().cpu().numpy()
            cv2.imwrite(os.path.join("./output/image/" , imagefileperfix + '.png' ), imageout*255)


if __name__ == '__main__':

    # Parameter
    cfg = parse()

    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{cfg.gpu}' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    model = StructureParsingNet(cfg).to(device)
    test_dataset = DatasetGSSU(cfg)

    test_loader = Data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    num_workers=cfg.num_workers,
                                    shuffle=False)
    test(model, test_loader, cfg, device)