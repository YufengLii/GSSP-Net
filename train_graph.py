import os
import numpy as np
from tqdm import tqdm
import random
import shutil
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from model.StructureParsing import StructureParsingNet
from config.cfg import parse
from dataset.SSUdataset import DatasetGSSU
import torch.nn.functional as F
from util.rendering import graph_link_render_batch


def train_graph(model, loader, cfg, device):

    linkglocallossF = torch.nn.MSELoss(reduction='sum')

    if cfg.last_epoch != -1:
        
        model_dict = model.state_dict()

        checkpoint_file = os.path.join(cfg.checkpointdir,
                                       f'junc-{cfg.last_epoch:03d}.pkl')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = checkpoint['model_state_dict']
        
        pretrained_dict_2 = {}
        for k, v in pretrained_dict.items():
            if "StructureParsingGCN" in k:
                continue
            elif "p_enc_2d_model_sum" in k:
                continue
            else:
                pretrained_dict_2[k] = v
        model_dict.update(pretrained_dict_2) 
        model.load_state_dict(model_dict)

    if cfg.Train_backbone:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=cfg.lr_graph,
                                    weight_decay=cfg.weight_decay)

    else:

        for param in model.ShapeBackbone.parameters():
            param.requires_grad = False
        # for param in model.SemanticBackbone.parameters():
        #     param.requires_grad = False    
        optimizer = torch.optim.Adam(model.StructureParsingGCN.parameters(),
                                    lr=cfg.lr_graph,
                                    weight_decay=cfg.weight_decay,
                                    amsgrad=True)
        
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 4], gamma=0.5)
    # Summary
    if os.path.exists(cfg.log_graph_path):
        shutil.rmtree(cfg.log_graph_path)

    writer = SummaryWriter(cfg.log_graph_path)
    model.to(device)
    step = 1

    for epoch in range(0, cfg.num_graph_epochs):
        # Train
        model.train()
        for images, gt_annos, gt_jmap, gt_joff, filepaths in tqdm(
                loader, desc='train: '):

            images = images.to(device)
            gt_annos = gt_annos.to(device)
            junc_keep, graph_edge, query_edge_index, link_preds = model(images)
            
            link_pred_sort_index = [
                torch.argsort(link_preds[index_i], descending=True)
                for index_i in range(len(link_preds))
            ]

            render_index = []
            for index_i in range(len(link_preds)):
                if link_preds[index_i].shape[0] >= cfg.link_render_number:

                    render_index.append(link_pred_sort_index[index_i]
                                        [0:cfg.link_render_number])
                else:
                    render_index.append(link_pred_sort_index[index_i])

            render_points_batch = []
            render_alpha_batch = []

            for index_i in range(len(link_preds)):
                sample_bi_j = junc_keep[index_i]
                query_edge_index_j = query_edge_index[index_i]
                render_index_j = render_index[index_i]

                render_points = [[
                    sample_bi_j[query_edge_index_j[item][1]],
                    sample_bi_j[query_edge_index_j[item][0]]
                ] for item in render_index_j]
                # may need adjust when draw
                render_points = torch.tensor(np.array(render_points) *
                                             4.0).to(device)
                render_alpha = link_preds[index_i][render_index_j]
                render_points_batch.append(render_points)
                render_alpha_batch.append(render_alpha)

            rendered_result, _ = graph_link_render_batch(render_points_batch,
                                                         render_alpha_batch,
                                                         force_cpu=False,
                                                         canvas_size=512,
                                                         colors=None,
                                                         withDraw=False,
                                                         step=step)

            link_global_loss = (rendered_result-gt_annos).pow(2).sum() / len(link_preds)
            # link_global_loss = linkglocallossF(rendered_result, gt_annos) 
            # loss_local = BCELoss(reduction="mean")(torch.cat(link_preds), link_lable)
            loss =  link_global_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.log_graph_print_freq == 0:

                lr = scheduler.get_last_lr()[0]

                print( 'epoch: %d/%d | loss: %6f  | link loss: %6f | lr: %6f '
                    % (epoch, cfg.num_junc_epochs, loss.item(), link_global_loss.item(), lr))
                
                writer.add_scalar('link_loss', link_global_loss, step)
                # writer.add_scalar('link_local_loss', loss_local, step)
                writer.add_image("gt sketckes edge map", gt_annos[0], step)
                writer.add_image("rasterized graph", rendered_result[0], step)
                writer.add_scalar('lr', lr, step)
            step += 1

        checkpoint_file = os.path.join(cfg.checkpointdir,
                                       f'graph-{epoch:03d}.pkl')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_file)

        scheduler.step()
    writer.close()


if __name__ == '__main__':

    # Parameter
    cfg = parse()
    os.makedirs(cfg.checkpointdir, exist_ok=True)
    os.makedirs(cfg.log_graph_path, exist_ok=True)
    # Use GPU or CPU
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
    train_dataset = DatasetGSSU(cfg)

    train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )
    
    train_graph(model, train_loader, cfg, device)
