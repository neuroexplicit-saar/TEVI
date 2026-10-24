import torch

import sparse_autoencoder
import sparse_autoencoder.default_config
import sparse_autoencoder.sae_trainer
import sparse_autoencoder.kernels
from mgap import utils
import os
from mgap import config
from tqdm import tqdm
import wandb
import time
from mgap import metrics
import os.path as osp
import torch
from torch.utils.data import DataLoader, TensorDataset
from mgap import config, filtering_architectures
import argparse
from open_clip_train.params import parse_args
from open_clip_train.distributed import init_distributed_device
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--scaler-growth-interval", type=int, default=8_000_000, help="Growth interval for grad scaler")
parser.add_argument("--eval-on-imagenet", action="store_true", default=False)
args = parse_args(args=None, parser=parser)
device = init_distributed_device(args)
utils.common_init(args)

full_ckpt_path = os.path.join(args.checkpoint_path, "checkpoint_final.pt")
if os.path.exists(full_ckpt_path):
    print(f"Checkpoint already exists at {full_ckpt_path}. Exiting...")
    sys.exit(0)


if not args.use_pretrained_sae and not args.resume_from_this_checkpoint:
    print("WARNING: Training SAE from scratch!")
filter_obj = filtering_architectures.SAEFilterTransform(args, device=torch.device(args.device))

if args.use_fixed_decoder_weights:
    decoder_weights = torch.load(os.path.join(args.load_clip_path, f"{args.decoder_weights_vocab}_vocab_embeddings.pt")).to(args.device)
    fixed_vocab_dim = decoder_weights.size(0)  # 49405 for full vocab, 26 for MAD

else:
    decoder_weights = None
    fixed_vocab_dim = None

ae = filter_obj.ae
cfg = filter_obj.cfg

ae.train()


if args.use_fixed_decoder_weights:
    args.expansion_factor = ae.decoder.weight.size(1) // decoder_weights.size(0)
    ae.decoder.weight.data = decoder_weights.T.repeat(1, args.expansion_factor)
    ae.decoder.weight.requires_grad = False
    # Seems to be needed as per the constructor
    ae.decoder.weight.data = ae.decoder.weight.data.T.contiguous().T
    sparse_autoencoder.sae_trainer.unit_norm_decoder_(ae)


if args.wandb:
    wandb_log_dir = args.log_base_path
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.environ["WANDB__SERVICE_WAIT"] = "500"
    logger = sparse_autoencoder.sae_trainer.Logger(
        entity=args.wandb_entity,
        project=args.wandb_project_name,
        name=args.sae_config_name + "-" + args.load_name,
        config=vars(args),
        dir=wandb_log_dir,
        mode=args.wandb_mode,
    )


num_train_pieces = 1
train_img_features_and_attributes = torch.load(osp.join(args.load_clip_path, 'train_features_and_attributes.pt'), map_location='cpu')
train_img_features = train_img_features_and_attributes['features']
train_img_attributes = train_img_features_and_attributes['attribute_lists']

val_img_features_and_attributes = torch.load(osp.join(args.load_clip_path, 'val_features_and_attributes.pt'))
val_img_features = val_img_features_and_attributes['features']
val_img_attributes = val_img_features_and_attributes['attribute_lists']

train_dataset = TensorDataset(train_img_features, train_img_attributes)
val_dataset = TensorDataset(val_img_features, val_img_attributes)
train_loader = DataLoader(train_dataset, batch_size=args.train_sae_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.train_sae_batch_size, shuffle=False)

scaler = torch.cuda.amp.GradScaler(growth_interval=args.scaler_growth_interval)

autocast_ctx_manager = torch.cuda.amp.autocast()

param_list = []
param_list += [{'params': list(ae.parameters()), 'lr': args.train_sae_lr}]

orig_metric_dict = {}
recon_metric_dict = {}
for attr in config.mad_decoder_name_list_cat_wise_literal:
    zeroshot_weight = torch.load(os.path.join(args.load_clip_path, f"{attr}_embeddings.pt")).to(args.device)
    orig_metric_dict[attr] = metrics.MADZeroShot(attr, zeroshot_weight).to(args.device)
for attr in config.mad_decoder_name_list_cat_wise_literal:
    zeroshot_weight = torch.load(os.path.join(args.load_clip_path, f"{attr}_embeddings.pt")).to(args.device)
    recon_metric_dict[attr] = metrics.MADZeroShot(attr, zeroshot_weight).to(args.device)

if num_train_pieces > 1:
    actual_epochs = args.train_sae_epochs * num_train_pieces
    train_order = torch.randperm(num_train_pieces)
    train_piece_idx = 0
else:
    actual_epochs = args.train_sae_epochs


if args.optimizer_name == 'Adam':
    opt = torch.optim.Adam(param_list, eps=cfg.eps, fused=True)
elif args.optimizer_name == 'AdamW':
    opt = torch.optim.AdamW(param_list, eps=cfg.eps, fused=True)

mse_scale = (
    1 / ((train_img_features.float().mean(dim=0) -
        train_img_features.float()) ** 2).mean()
)
mse_scale = mse_scale.item()


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_epochs, eta_min=0)

    
start_epoch = 0 


for current_epoch in tqdm(range(start_epoch, actual_epochs)):
    completed_epoch = current_epoch + 1
    if num_train_pieces > 1:
        del train_img_features, train_dataset, train_loader
        if train_piece_idx >= num_train_pieces:
            train_order = torch.randperm(num_train_pieces)
            train_piece_idx = 0
        train_img_features = torch.load(osp.join(fname_dir, train_fnames[train_order[train_piece_idx]]), map_location='cpu')
        print(f"Current Epoch: {current_epoch}, Train Piece Idx: {train_piece_idx}, Train Order Idx: {train_order[train_piece_idx]}, Num Samples: {len(train_img_features)}")
        train_piece_idx += 1
        train_dataset = TensorDataset(train_img_features)
        train_loader = DataLoader(train_dataset, batch_size=args.train_sae_batch_size, shuffle=True)

    start_epoch = time.time()
    batch_total_loss = 0
    batch_sae_loss = 0
    batch_bce_loss = 0
    total_count = 0
    # img_data: tuple of (img_features, img_attributes) for MAD
    # img_data: tuple of (img_features) for CC3M
    # img_data: dict for MAD with use_fixed_latent_meanings
    for batch_idx, img_data in enumerate(train_loader):
        img_features = img_data[0]
        img_features = img_features.to(args.device)

        opt.zero_grad()

        recons, info = ae(img_features)
        bce_loss = torch.tensor(0.0)

        sae_loss = utils.sae_loss_fn(
            ae, img_features, recons, info, logger, cfg, mse_scale, split="train")[0]

        total_loss = sae_loss + args.latent_zero_loss_lambda * bce_loss
        batch_total_loss += (total_loss.item() * img_features.size(0))
        batch_sae_loss += (sae_loss.item() * img_features.size(0))
        batch_bce_loss += (bce_loss.item() * img_features.size(0))
        total_count += img_features.size(0)

        total_loss = scaler.scale(total_loss)
        total_loss.backward()

        if not args.use_fixed_decoder_weights:
            sparse_autoencoder.sae_trainer.unit_norm_decoder_(ae)
            sparse_autoencoder.sae_trainer.unit_norm_decoder_grad_adjustment_(ae)


        # set the gradients of the sae decoder first few dimensions as zero 


        # take step with optimizer
        scaler.step(opt)
        scaler.update()

    batch_total_loss /= total_count
    batch_sae_loss /= total_count
    batch_bce_loss /= total_count

    if args.wandb:
        train_wandb_dict = {}
        train_wandb_dict.update({"train/bce_loss": batch_bce_loss})
        train_wandb_dict.update({"train/sae_loss": batch_sae_loss})
        train_wandb_dict.update({"train/total_loss": batch_total_loss})
        train_wandb_dict.update({"train/loss_scale": scaler.get_scale()})
        train_wandb_dict.update({"train/epoch": current_epoch})
        train_wandb_dict.update({"train/sae_lr": opt.param_groups[0]['lr']})
        wandb.log(train_wandb_dict, commit=False)

    print(f"Epoch: {current_epoch}, Time taken: {time.time()-start_epoch}, Loss: {batch_total_loss}, SAE Loss: {batch_sae_loss}, BCE Loss: {batch_bce_loss}")
    scheduler.step()


    ############################################
    evaluate_this_epoch = (args.val_frequency != 0 and (current_epoch // num_train_pieces) % args.val_frequency == 0 and (current_epoch // num_train_pieces) != 0)
    if evaluate_this_epoch:
        print("Validating...")
        vstart = time.time()
        val_wandb_dict = {}
        with torch.no_grad():
            val_total_loss = 0
            val_sae_loss = 0
            val_bce_loss = 0
            val_total_count = 0
            for batch_idx, val_data in enumerate(val_loader):
                val_img_features, val_img_attributes = val_data[0], val_data[1]
                val_img_attributes = val_img_attributes.to(args.device)
                val_img_features = val_img_features.to(args.device)

                recons, info = ae(val_img_features)
                bce_loss = torch.tensor(0.0)

                for idx, attr in enumerate(orig_metric_dict.keys()):
                    orig_metric_dict[attr].update(val_img_features, val_img_attributes[:, idx])
                    recon_metric_dict[attr].update(recons, val_img_attributes[:, idx])

                sae_loss = utils.sae_loss_fn(
                    ae, val_img_features, recons, info, logger, cfg, mse_scale, split="train_val")[0]
                total_loss = sae_loss + args.latent_zero_loss_lambda * bce_loss
                val_total_loss += (total_loss.item() * val_img_features.size(0))
                val_sae_loss += (sae_loss.item() * val_img_features.size(0))
                val_bce_loss += (bce_loss.item() * val_img_features.size(0))
                val_total_count += val_img_features.size(0)

            val_total_loss /= val_total_count
            val_sae_loss /= val_total_count
            val_bce_loss /= val_total_count

            ###########################
            print(f'Val Epoch: {current_epoch+1}, Loss: {val_total_loss}, SAE Loss: {val_sae_loss}, BCE Loss: {val_bce_loss}')
            total_orig_accuracy = 0.0
            total_recon_accuracy = 0.0
            for attr in orig_metric_dict.keys():
                orig_metric_vals = orig_metric_dict[attr].compute()
                recon_metric_vals = recon_metric_dict[attr].compute()
                print(f"Attribute: {attr}, Original: {orig_metric_vals['accuracy']}, Reconstructed: {recon_metric_vals['accuracy']}")
                if args.wandb:
                    val_wandb_dict.update({f"val-zeroshot-orig/{attr}_accuracy": orig_metric_vals['accuracy']})
                    val_wandb_dict.update({f"val-zeroshot-recon/{attr}_accuracy": recon_metric_vals['accuracy']})
                total_orig_accuracy += orig_metric_vals['accuracy']
                total_recon_accuracy += recon_metric_vals['accuracy']
            total_orig_accuracy /= len(orig_metric_dict.keys())
            total_recon_accuracy /= len(orig_metric_dict.keys())
            print(f"Mean Original Accuracy: {total_orig_accuracy}, Mean Reconstructed Accuracy: {total_recon_accuracy}")
            if args.wandb:
                val_wandb_dict.update({"val/sae_loss": val_sae_loss})
                val_wandb_dict.update({"val/bce_loss": val_bce_loss})
                val_wandb_dict.update({"val/total_loss": val_total_loss})
                val_wandb_dict.update({"val/mean_orig_accuracy": total_orig_accuracy})
                val_wandb_dict.update({"val/mean_recon_accuracy": total_recon_accuracy})

        if args.wandb:
            wandb.log(val_wandb_dict, commit=False)

        print(f"Validation done! - time: taken: {time.time()-vstart}")

    logger.dumpkvs()

    if (args.save_frequency != 0 and (completed_epoch % (num_train_pieces * args.save_frequency)) == 0):
        utils.save_ckpt(args, args.checkpoint_path, completed_epoch, sae_model=ae, clip_model=None, linear_transform=None, optimizer=opt, scaler=scaler, mse_scaler=mse_scale)

utils.save_ckpt(args, args.checkpoint_path, "final", sae_model=ae, clip_model=None, linear_transform=None, optimizer=opt, scaler=scaler, mse_scaler=mse_scale)

