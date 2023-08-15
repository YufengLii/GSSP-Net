import os
import numpy as np
from tqdm import tqdm
import random
import shutil
import torch
import torch.utils.data as Data
from util.losses import weighted_l1_loss_off, bdcn_loss2, cross_entropy_loss_RCF
from tensorboardX import SummaryWriter
from model.StructureParsing import StructureParsingNet
from config.cfg import parse
from dataset.SSUdataset import DatasetGSSU
import torch.nn.functional as F

def train_junc(model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(
        model.parameters(),lr=cfg.lr_junc, weight_decay=cfg.weight_decay
    )

    if cfg.last_epoch != -1:
        checkpoint_file = os.path.join(
            cfg.checkpointdir, f"junc-{cfg.last_epoch:03d}.pkl"
        )
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15], gamma = 0.1
    )

    # Summary
    if os.path.exists(cfg.log_junc_path):
        shutil.rmtree(cfg.log_junc_path)
    writer = SummaryWriter(cfg.log_junc_path)
    model.to(device)
    
    step = 1
    for epoch in range(cfg.last_epoch + 1, cfg.num_junc_epochs):
        # Train
        model.train()
        for images, gt_annos, gt_jmap, gt_joff, filepaths in tqdm(loader, desc="train: "):

            images = images.to(device)
            gt_annos = gt_annos.to(device)
            gt_jmap = gt_jmap.to(device)
            gt_joff = gt_joff.to(device)

            ResultEdgeMaps, ResultJuncMaps = model(images)

            gt_jmap = gt_jmap.type_as(ResultJuncMaps["jmap"])
            gt_joff = gt_joff.type_as(ResultJuncMaps["joff"])

            jmap_preds = ResultJuncMaps["jmap"]
            joff_preds = ResultJuncMaps["joff"]

            jmap_loss = F.binary_cross_entropy(jmap_preds, gt_jmap)
            joff_loss = weighted_l1_loss_off(joff_preds, gt_joff, gt_jmap)

            edge_loss = 0
            if cfg.EdgeBackboneName == "dexined":
                edge_loss = sum(
                    [
                        bdcn_loss2(preds, gt_annos, l_w)
                        for preds, l_w in zip(ResultEdgeMaps, [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3])
                    ]
                )

            # elif cfg.EdgeBackboneName == "pidinet":
            #     for o in ResultEdgeMaps:
            #         edge_loss += cross_entropy_loss_RCF(o, gt_annos, 1.1)

            loss = jmap_loss + joff_loss + edge_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.log_junc_print_freq == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    "epoch: %d/%d | loss: %6f | Jmap loss: %6f | joff loss: %6f | edge loss: %6f | lr: %6f "
                    % (
                        epoch,
                        cfg.num_junc_epochs,
                        loss.item(),
                        jmap_loss.item(),
                        joff_loss.item(),
                        edge_loss.item(),
                        lr,
                    )
                )
                
                ResultEdgeMaps_fuse = ResultEdgeMaps[-1] # 8 1 512 512
                result_fuse_0 = torch.squeeze(ResultEdgeMaps_fuse[0,:,:,:]).detach().cpu().numpy()
                
                writer.add_image("edge", result_fuse_0, step, dataformats='HW')
                writer.add_scalar("loss", loss, step)
                writer.add_scalar("jmap_loss", jmap_loss, step)
                writer.add_scalar("joff_loss", joff_loss, step)
                writer.add_scalar("lmap_loss", joff_loss, step)
                writer.add_scalar("edge_loss", edge_loss, step)
                writer.add_scalar("lr", lr, step)

            step += 1
        checkpoint_file = os.path.join(cfg.checkpointdir, f"junc-{epoch:03d}.pkl")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_file,
        )

        scheduler.step()
    writer.close()


if __name__ == "__main__":

    # Parameter
    cfg = parse()
    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.gpu}" if use_gpu else "cpu")
    print("use_gpu: ", use_gpu)
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

    if cfg.train_flag == "junc":
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=cfg.train_batch_size,
            num_workers=cfg.num_workers,
            # collate_fn=my_collate_fn,
            shuffle=True,
        )
        train_junc(model, train_loader, cfg, device)
