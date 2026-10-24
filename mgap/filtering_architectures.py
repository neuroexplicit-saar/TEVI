import torch
import sparse_autoencoder
import sparse_autoencoder.default_config
import sparse_autoencoder.sae_trainer
import sparse_autoencoder.kernels
import os
from mgap import utils
import os.path as osp
from open_clip_train.file_utils import pt_load
import logging
import contextlib
from mgap import config
from open_clip_train.train import unwrap_model


"""
Architectures we have in mind:
- Identity
- SAE (TopK)
- Linear layer
- MLP
- AE
"""


def get_filtering_architecture(args, device=None, **kwargs):
    filter_architecture_dict = {
        "Identity": ConceptFilterTransform,
        "IdentityMaskable": IdentityMaskableFilterTransform,
        "SAE": SAEFilterTransform,
    }
    return filter_architecture_dict[args.filter_arch_name](args, device, **kwargs)


class ConceptFilterTransform:

    def __init__(self, args, device, **kwargs):
        """
        Constructor for initialization. Performs Identity filtering when this class is used.
        """
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.device = device


    def get_latents(self, embeddings, **kwargs):
        """
        Perform the transform and get the latent concepts.

        Args:
            embeddings (torch.Tensor): CLIP embeddings to transform, shape (n_images, clip_embedding_dim).

        Returns:
            dict: Info dictionary containing either latents or inds and vals. Latents with shape (n_images, n_latents) and inds and vals with shapes (n_images, k) and (n_images, k) respectively.
        """
        return {"inds": None, "vals": None, "latents": embeddings}

    def get_reconstructed_features(self, embeddings, info):
        """
        Reconstruct the embeddings from the latents.

        Args:
            embeddings (torch.Tensor): CLIP embeddings given as input, shape (n_images, clip_embedding_dim).
            info (dict): Info dictionary containing either latents or inds and vals. Latents with shape (n_images, n_latents) and inds and vals with shapes (n_images, k) and (n_images, k) respectively.

        Returns:
            torch.Tensor: Reconstructed embeddings from the latents in info.
        """
        return info["latents"]

    def get_loss(self, embeddings, reconstruction=None, info=None):
        """
        Returns any losses for training the transform layers.

        Args:
            embeddings (torch.Tensor): CLIP embeddings given as input, shape (n_images, clip_embedding_dim).
            reconstruction (torch.Tensor, optional): Known reconstruction if forward pass already done. Defaults to None. Shape (n_images, clip_embedding_dim).
            info (dict, optional): Known info dict if encoder forward pass already done. Defaults to None. Latents with shape (n_images, n_latents) and inds and vals with shapes (n_images, k) and (n_images, k) respectively.

        Returns:
            tuple: Tuple of loss and dictionary containing losses to be logged
        """
        return 0, {}

    def get_optimizer_param_list(self):
        """
        Returns the parameters to be added to the optimizer when training.

        Returns:
            list: List of optimization parameters for training the filter. Returns empty list when in eval mode.
        """
        return []

    def post_backward_operation(self):
        """
        Perform any operations after the backward pass. For example, setting the decoder weights to be unit normalized
        in an SAE.
        """
        pass

    def unpack_info(self, info):
        """
        Returns latents, inds, and vals from info dict, and None if key not present.

        Args:
            info (dict): Info dictionary containing either latents or inds and vals. Latents with shape (n_images, n_latents) and inds and vals with shapes (n_images, k) and (n_images, k) respectively.

        Returns:
            tuple: Tuple of latents, inds, and vals.
        """
        latents = info["latents"] if "latents" in info else None
        inds = info["inds"] if "inds" in info else None
        vals = info["vals"] if "vals" in info else None
        return latents, inds, vals

    def get_checkpoint_dict(self):
        """
        Returns the dictionary to be saved in the checkpoint. For example, for an SAE the SAE weights.

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: Dictionary of all the parameters to be saved in the checkpoint.
        """
        return {}

    def get_metrics(self, embeddings, reconstruction, info):
        """
        Returns any filter-specific metrics to be logged.

        Args:
            embeddings (torch.Tensor): CLIP embeddings given as input, shape (n_images, clip_embedding_dim).
            reconstruction (torch.Tensor): Reconstructed of the embeddings, shape (n_images, clip_embedding_dim).
            info (dict): Info dictionary containing either latents or inds and vals. Latents with shape (n_images, n_latents) and inds and vals with shapes (n_images, k) and (n_images, k) respectively.

        Returns:
            dict: Dictionary of metrics to be logged.
        """
        return {}

    def get_wandb_name(self):
        """
        Returns the name to be used in wandb. Not sure if used currently.

        Returns:
            str: Name to be used in wandb.
        """
        return "Identity"

    def load_checkpoint(self, checkpoint):
        pass

    def init_distributed(self):
        pass

    def freeze_parameters(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class IdentityMaskableFilterTransform(ConceptFilterTransform):
    """
    Identity transform that additionally exposes ``embed_dim`` and ``latent_dim`` so a
    LearntMask (or similar) algorithm can build a mask operating directly on the CLIP
    embeddings (no SAE in between).

    ``get_latents`` (inherited) returns the CLIP embeddings under the ``latents`` key and
    ``get_reconstructed_features`` (inherited) returns them unchanged, so the mask is
    applied element-wise on the CLIP embedding dimensions:  ``clip_embeddings * mask``.
    """

    def __init__(self, args, device, **kwargs):
        super().__init__(args, device, **kwargs)
        embed_dim = kwargs.get("embed_dim", None)
        if embed_dim is None:
            embed_dim = config.embed_dim_dict[args.model]
        # Mask lives in CLIP embedding space, so latent_dim == embed_dim.
        self.embed_dim = embed_dim
        self.latent_dim = embed_dim

    def get_wandb_name(self):
        return "IdentityMaskable"


class SAEFilterTransform(ConceptFilterTransform):

    def __init__(self, args, device, **kwargs):
        super().__init__(args, device, **kwargs)
        if args.use_fixed_decoder_weights or args.use_fixed_latent_meanings or args.use_fixed_vocab_dim:
            self.fixed_vocab_dim = 26
                
        else:
            decoder_weights = None
            self.fixed_vocab_dim = None

        cfg = sparse_autoencoder.default_config.Config()
        cfg = utils.set_cfg_params(
            cfg, args, fixed_vocab_dim=self.fixed_vocab_dim)
        ae, comms = utils.get_sae(cfg, args)
        self.ae = ae
        self.cfg = cfg
        self.mse_scale = None
        self.comms = comms

        if args.train_sae: 
            self.ae.train()


        if args.use_pretrained_sae and not args.random_sae:
            print(f"SAE checkpoint path, using {args.load_sae_checkpoint_path}")


            if os.path.exists(args.load_sae_checkpoint_path): # since we use the same script to dump clip features
                sae_checkpoint_dict = pt_load(args.load_sae_checkpoint_path)
                sae_state_dict = sae_checkpoint_dict["sae_model"]
        if args.use_pretrained_sae and not args.random_sae:
            self.ae.load_state_dict(sae_state_dict)
            logging.info(f"=> Loading pretrained SAE models from '{args.load_sae_checkpoint_path}'")
            print(f"=> Loading pretrained SAE models from '{args.load_sae_checkpoint_path}'")


        if self.mse_scale is None:
            if args.use_pretrained_sae and not args.random_sae and "mse_scaler" in sae_checkpoint_dict.keys():
                self.mse_scale = sae_checkpoint_dict["mse_scaler"]
            else:
                self.set_mse_scale_from_data()
                
        if args.train_sae and not args.use_pretrained_sae and not args.hybrid_sae_decoder:
            sparse_autoencoder.sae_trainer.init_from_data_(
                self.ae, self.train_img_features.clone(), comms)
        else:
            self.ae.eval()

        self.embed_dim = self.ae.encoder.weight.shape[1]
        self.latent_dim = self.ae.encoder.weight.shape[0]

        
    def set_encoder_weight_from_data(self): 
        # used when using hybrid sae decoder
        sparse_autoencoder.sae_trainer.init_from_data_(
                self.ae, self.train_img_features.clone(), self.comms)


    def init_distributed(self):
        if self.args.distributed and not self.args.horovod:
            self.ae = torch.nn.parallel.DistributedDataParallel(
                self.ae, device_ids=[self.device], **self.kwargs)

    def set_mse_scale_from_data(self):
        """
        Computes the MSE scale from the training data for the given SAE config, and sets the mse_scale attribute.
        The training features must be present in the SAE directory.
        """
        self.train_img_features = torch.load(osp.join(
            self.args.load_clip_path, "train_features_and_attributes.pt"))["features"]
        print(
            f"Load train img features path: {osp.join(self.args.load_clip_path, 'train_features_and_attributes.pt')}")
        mse_scale = (
            1 / ((self.train_img_features.float().mean(dim=0) - self.train_img_features.float()) ** 2).mean()
        )
        self.mse_scale = mse_scale.item()

    def post_backward_operation(self):
        if self.args.train_sae:
            sparse_autoencoder.sae_trainer.unit_norm_decoder_(
                self.ae, unwrap=True)
            sparse_autoencoder.sae_trainer.unit_norm_decoder_grad_adjustment_(
                self.ae, unwrap=True)

    def get_latents(self, embeddings, **kwargs):
        _, info = self.ae(embeddings, get_only_latents=True)
        return info

    def get_reconstructed_features(self, embeddings, info):
        latents, inds, vals = self.unpack_info(info) # in dict name is latents_pre_act
        assert latents is not None or (inds is not None and vals is not None)
        if latents is not None:
            raise NotImplementedError
            reconstruction = latents @ self.ae.decoder.weight.T + self.ae.pre_bias
            return reconstruction
        else:
            reconstruction = self.ae(
                None, inds=inds, vals=vals, decode_only=True)
            return reconstruction

    def get_loss(self, embeddings, reconstruction=None, info=None, **kwargs):
        context = torch.no_grad() if not self.args.train_sae else contextlib.nullcontext()
        unwrapped_ae = unwrap_model(self.ae)
        with context:
            if info is None:
                info = self.get_latents(embeddings, get_only_latents=True)

            if reconstruction is None:
                reconstruction = self.get_reconstructed_features(
                    embeddings, info)

            latents, inds, vals = self.unpack_info(info)
            assert latents is not None or (
                inds is not None and vals is not None)
            term1 = self.mse_scale * \
                sparse_autoencoder.kernels.mse(reconstruction, embeddings)
            term2 = self.cfg.auxk_coef * sparse_autoencoder.kernels.normalized_mse(
                self.ae(None,
                        inds=info["auxk_inds"],
                        vals=info["auxk_vals"],
                        decode_only=True
                        ),
                embeddings - reconstruction.detach() + unwrapped_ae.pre_bias.detach(),
            ).nan_to_num(0)
            sae_loss = term1 + term2
            dict = {"sae_loss": sae_loss,
                    "recon_loss": term1, "maxk_recon_loss": term2}
        if not self.args.train_sae:
            sae_loss = 0
        return sae_loss, dict

    def get_part_reconstructed_features(self, inds, vals, latents_pre_act, txt_features_norm, cumsums):
        """
        Computes reconstructed features by matching dictionary vectors from the decoder with text features
        of the words in the caption, and only using these dictionary vectors for reconstruction. Used with the
        PartContra algorithm.

        Args:
            inds (torch.Tensor): Indices of TopK concepts for each input, shape (n_images, k).
            vals (torch.Tensor): Values of TopK concepts at the indices, shape (n_images, k).
            latents_pre_act (torch.Tensor): Full latent vector before ReLU, shape (n_images, latent_dim).
            txt_features_norm (torch.Tensor): Embeddings of all individual words in all captions, placed one after the other, shape (n_words, clip_embedding_dim).
            cumsums (torch.Tensor): Positions at which one caption ends and the next begins, shape (n_images+1,).

        Returns:
            dict: Dictionary containing embeddings with key "embeddings", mean of top cosine similarities with key "top_cos_mean", and number of dead nodes with key "num_dead_nodes".
        """

        embeddings = torch.zeros((len(inds), txt_features_norm.shape[1]))
        top_cos_total = 0.0
        post_relu_acts = torch.relu(latents_pre_act)  # to check dead nodes
        num_dead_nodes = 0.0
        ###########
        for img_idx in range(len(inds)):
            # selected_bias= args.ae.pre_bias[:, inds[]]
            with torch.no_grad():
                selected_dictionary_vectors = self.ae.decoder.weight[:,
                                                                     inds[img_idx]]
                selected_txt_features = txt_features_norm[cumsums[img_idx]: cumsums[img_idx + 1]]
                if len(selected_txt_features) == 0:
                    continue
                cosine_similarities = selected_txt_features @ selected_dictionary_vectors
                # get topk cosine sim and their indices

                top_cos = torch.topk(cosine_similarities,
                                     k=self.args.num_dicvec_per_word, dim=1)
                # get the indices of the topk cosine sim
                top_inds = top_cos.indices
                # <=num_dicvec_per_word* n_words
                uniq_top_inds = torch.unique(top_inds.flatten())
                uniq_vals = vals[img_idx][uniq_top_inds]
                top_inds_decoder = inds[img_idx][uniq_top_inds]

                concept_strength = post_relu_acts[:, top_inds_decoder]
                num_dead_nodes += (concept_strength.sum(dim=0).eq(0).sum())

            # get the corresponding dictionary vectors

            top_dictionary_vectors = self.ae.decoder.weight[:,
                                                            top_inds_decoder]
            embeddings[img_idx] = (
                uniq_vals @ top_dictionary_vectors.T) + self.ae.pre_bias
            top_cos_total += top_cos.values.mean()

        top_cos_mean = top_cos_total / len(inds)
        embeddings = embeddings.to(self.args.device)

        return_dict = {"embeddings": embeddings,
                       "top_cos_mean": top_cos_mean, "num_dead_nodes": num_dead_nodes}

        return return_dict

    def get_optimizer_param_list(self):
        if self.args.train_sae:
            param_list = [{'params': list(self.ae.parameters(
            )), 'lr': self.args.train_sae_lr, "eps": self.cfg.eps, "fused": True, "name": "filter_arch"}]
        else:
            param_list = []
        return param_list

    def get_checkpoint_dict(self):
        if not self.args.train_sae:
            return {}
        else:
            return {"sae_model": self.ae.state_dict(), "sae_config": self.args.sae_config_name, "cfg": self.cfg}

    def load_checkpoint(self, checkpoint):
        self.ae.load_state_dict(checkpoint["sae_model"])

    def get_wandb_name(self):
        return "SAE"

    def eval(self):
        self.ae.eval()

    def train(self):
        self.ae.train()

    def freeze_parameters(self):
        for param in self.ae.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.ae.parameters():
            param.requires_grad = True