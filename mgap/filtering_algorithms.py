import torch.nn as nn
import torch
from mgap import utils
import functools
import numpy as np

"""
Algorithms:
- Identity: No filtering
- Part Contra: Filter based on subset of words
- Recon Contra: Filter based on subset that reconstructs closest to the other modality
- Learnt Masking: Learn a mask to filter concepts
"""


def get_filtering_algorithm(args, device=None, **kwargs):
    filter_algorithms_dict = {
        "Identity": ConceptFilter,
        "PartContra": PartContraConceptFilter,
        "ReconContra": ReconContraConceptFilter,
        "LearntMask": LearntMaskConceptFilter,
        "GTMask": GTMaskConceptFilter,
        "SimpleMLP": SimpleMLP,
        "CrossAttention": CrossAttention,
    }
    return filter_algorithms_dict[args.filter_algo_name](args, device, **kwargs)


class ConceptFilter(torch.nn.Module):

    def __init__(self, args, device, **kwargs):
        """
        Constructor for initialization. Performs Identity filtering when this class is used.
        """
        super().__init__()
        self.filter_arch = args.filter_arch_name
        self.args = args
        self.device = device
        self.kwargs = kwargs

    def post_backward_operation(self):
        """
        Performs any operations needed after backward pass, for example setting the SAE decoder weights to unit norm.
        """
        self.filter_obj.post_backward_operation()

    def get_checkpoint_dict(self):
        """
        Returns a dictionary containing parameters to be saved in the checkpoint.

        Returns:
            dict: Dictionary containing parameters to be saved in the checkpoint.
        """
        return self.filter_obj.get_checkpoint_dict()

    def set_model(self, model, tokenizer, device):
        """
        Sets the model, tokenizer, and device. Not in the constructor since the model is not available at the time of initialization.

        Args:
            model (torch.nn.Module): CLIP model.
            tokenizer (torch.nn.Module): CLIP tokenizer.
            device (torch.device): Device to run the model on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def set_filtering_architecture(self, filter_obj):
        """
        Sets the filtering architecture, e.g. SAE.

        Args:
            filter_obj (ConceptFilterTransform): Filtering architecture object.
        """
        self.filter_obj = filter_obj

    def get_loss(self, imgs, texts, img_embeddings, text_embeddings, sae_outputs_dict):
        """
        Returns the additional losses from the filtering algorithm and architecture. Multiplies the loss with the appropriate lambda
        value so it can be directly added to the CLIP loss.

        Args:
            imgs (torch.Tensor): Images used for CLIP training, shape (batch_size, 3, 224, 224).
            texts (list): Captions used for CLIP training, length batch_size.
            img_embeddings (torch.Tensor): CLIP image embeddings, shape (batch_size, clip_embedding_dim).
            text_embeddings (torch.Tensor): CLIP text embeddings, shape (batch_size, clip_embedding_dim).
            sae_outputs_dict (dict): Dictionary containing the reconstruction and info values
            from a forward pass done when computing filtered embeddings.

        Returns:
            tuple: Tuple of loss value to be added scaled with lambda, and dictionary of individual losses to be logged.
        """
        loss, loss_dict = self.filter_obj.get_loss(
            img_embeddings, sae_outputs_dict["reconstruction"], sae_outputs_dict["info"])
        return self.args.contra_loss_lambda * loss, loss_dict

    def get_optimizer_param_list(self):
        """
        Returns the additional parameters to be added to the optimizer when training. Empty list when in eval mode.

        Returns:
            list: List of dictionaries containing parameters to be added to the optimizer.
        """
        return self.filter_obj.get_optimizer_param_list()

    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        """
        Returns the filtered image embeddings using the filtering architecture.

        Args:
            imgs (torch.Tensor): Images used for CLIP training, shape (batch_size, 3, 224, 224).
            texts (list): Captions used for CLIP training, length batch_size.
            img_embeddings (torch.Tensor): CLIP image embeddings, shape (batch_size, clip_embedding_dim).
            text_embeddings (torch.Tensor): CLIP text embeddings, shape (batch_size, clip_embedding_dim).

        Returns:
            tuple: Filtered image embeddings, and dictionary containing reconstruction and info.
        """
        info = self.filter_obj.get_latents(img_embeddings)
        sae_outputs_dict = {"info": info, "reconstruction": None}
        if "inds" in info:
            info_updated = {"inds": info["inds"], "vals": info["vals"], "latents_pre_act": info["latents_pre_act"], "mask": torch.ones_like(
                info["latents_pre_act"]).to(self.device)}
        else:
            info_updated = info
        sae_outputs_dict["info_updated"] = info_updated
        recon_img_embeddings = self.filter_obj.get_reconstructed_features(
            img_embeddings, info_updated)
        return recon_img_embeddings, sae_outputs_dict

    def get_wandb_name(self):
        """
        Returns the name to be used for logging in Weights and Biases. Combination of algo and architecture name.

        Returns:
            str: Name to be used for logging in Weights and Biases.
        """
        return self.args.filter_algo_name + "_" + self.args.filter_arch_name

    def get_metrics(self, imgs, texts, img_embeddings, text_embeddings):
        """
        Returns additional metrics to be logged when training and validating.

        Args:
            imgs (torch.Tensor): Images used for CLIP training, shape (batch_size, 3, 224, 224).
            texts (list): Captions used for CLIP training, length batch_size.
            img_embeddings (torch.Tensor): CLIP image embeddings, shape (batch_size, clip_embedding_dim).
            text_embeddings (torch.Tensor): CLIP text embeddings, shape (batch_size, clip_embedding_dim).
        """
        return {}

    def get_latents(self, embeddings):
        return self.filter_obj.get_latents(embeddings)

    def get_reconstructed_features(self, embeddings, info):
        return self.filter_obj.get_reconstructed_features(embeddings, info)

    def init_distributed(self):
        self.filter_obj.init_distributed()


class PartContraConceptFilter(ConceptFilter):

    def __init__(self, args):
        super().__init__(args)

    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings):
        info = self.filter_obj.get_latents(img_embeddings)
        sae_outputs_dict = {"info": info, "reconstruction": None}
        with torch.no_grad():
            txt_features_norm_word_level, cumsums = self.get_parts_of_caption_features(
                texts, self.model, self.tokenizer)
        return_dict = self.get_part_reconstructed_features(
            info['inds'], info['vals'], info['latents_pre_act'], txt_features_norm_word_level, cumsums)
        part_recon_image_features_unnorm = return_dict['embeddings']
        return part_recon_image_features_unnorm, sae_outputs_dict

    def get_parts_of_caption_features(self, train_caption, model, tokenizer):
        filtered_words = []
        for caption in train_caption:
            list_of_words = caption.split("-")
            filtered_words.append(list_of_words)
        tokenized_words = [(tokenizer(individual_words), len(individual_words))
                           for individual_words in filtered_words]
        token_list, word_len_list = zip(*tokenized_words)
        cumsums = torch.tensor(
            [0] + torch.tensor(word_len_list).cumsum(dim=0).tolist())
        token_list = torch.cat(token_list).to(self.args.device)
        with self.args.autocast():
            txt_features = model.encode_text(token_list).float()
            txt_features_norm = txt_features / \
                txt_features.norm(dim=-1, keepdim=True)
        return txt_features_norm, cumsums

    def get_part_reconstructed_features(self, inds, vals, latents_pre_act, txt_features_norm, cumsums):
        part_reconstruction = self.filter_obj.get_part_reconstructed_features(
            inds, vals, latents_pre_act, txt_features_norm, cumsums)
        return part_reconstruction

    def get_metrics(self):
        pass


class ReconContraConceptFilter(ConceptFilter):

    def __init__(self, args):
        super().__init__(args)

    def get_closest_subset_to_reconstruct(self, concept_strengths, dictionary_vectors, target_text_embedding, pre_bias):
        '''
        get_closest_subset_to_reconstruct: brute force
        # TODO: rewrite to handle beyond SAE
        '''
        # concept_strengths: (k,), e.g. (12, )
        # dictionary_vectors: (k, clip_embedding_dim), e.g. (12, 18)
        # target_text_embedding: (clip_embedding_dim,), e.g. (18, )
        input_embeddings = concept_strengths.unsqueeze(
            1) * dictionary_vectors  # (k, clip_embedding_dim), e.g. (12, 18)
        subset_matrix = utils.binary_matrix(len(concept_strengths)).to(
            self.args.device)  # (2**k, k), e.g. (4096, 12)
        # (2**k, clip_embedding_dim), e.g. (4096, 18)
        all_reconstructions = (subset_matrix @ input_embeddings) + pre_bias
        all_cossims = torch.nn.functional.cosine_similarity(
            all_reconstructions, target_text_embedding.unsqueeze(0), dim=1)  # (2**k,), e.g. (4096, )
        highest_cossim_idx = torch.argmax(all_cossims)
        index_selection_for_highest_cossim = subset_matrix[highest_cossim_idx]

        return index_selection_for_highest_cossim, all_reconstructions[highest_cossim_idx], all_cossims[highest_cossim_idx]

    def get_part_reconstructed_features(self, inds, vals, ae, text_embeddings_unnorm):
        '''
        get_all_closest_subset_to_reconstruct: brute force
        '''
        all_reconstructions = torch.zeros(
            (len(vals), ae.decoder.weight.shape[0]))
        for img_idx in range(len(vals)):
            all_reconstructions[img_idx] = self.get_closest_subset_to_reconstruct(
                vals[img_idx], ae.decoder.weight[:, inds[img_idx]].T, text_embeddings_unnorm[img_idx], ae.pre_bias)[1]
        return_dict = {"embeddings": all_reconstructions}
        return return_dict

    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings):
        info = self.filter_obj.get_latents(img_embeddings)
        reconstruction = self.filter_obj.get_reconstructed_features(
            img_embeddings, info)
        sae_outputs_dict = {"info": info, "reconstruction": reconstruction}

        return_dict = self.get_part_reconstructed_features(
            info['inds'], info['vals'], self.filter_obj, text_embeddings)
        part_recon_image_features_unnorm = return_dict['embeddings']

        return part_recon_image_features_unnorm, sae_outputs_dict

    def get_metrics(self):
        pass


class LearntMaskConceptFilter(ConceptFilter):

    def __init__(self, args, device, **kwargs):
        super().__init__(args, device, **kwargs)


    # https://stackoverflow.com/a/49433937/


    def init_weights(self, m, bias_fill=0.0):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:  # Ensure bias exists
                m.bias.data.fill_(bias_fill)

    def set_filtering_architecture(self, filter_obj):
        super().set_filtering_architecture(filter_obj)

        layer_list = [
            torch.nn.Linear(self.filter_obj.embed_dim,
                            self.filter_obj.latent_dim),
        ]

        if self.args.n_layers > 1:
            for _ in range(self.args.n_layers - 1):
                layer_list.extend([
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.filter_obj.latent_dim,
                                    self.filter_obj.latent_dim)
                ])

        # repeat entries of the list depending on a int 'c'
        # layer_list = [item for sublist in [[layer sfor layer in layer_list] for _ in range(self.filter_obj.c)] for item in sublist]
        self.masking_layer = torch.nn.Sequential(
            *layer_list).to(self.args.device)
        self.masking_layer.apply(functools.partial(
            self.init_weights, bias_fill=self.args.sigmoid_bias_fill))

    def init_distributed(self):
        self.filter_obj.init_distributed()
        if self.args.distributed and not self.args.horovod:
            self.masking_layer = torch.nn.parallel.DistributedDataParallel(
                self.masking_layer, device_ids=[self.device], **self.kwargs)
            self.null_predictor = torch.nn.parallel.DistributedDataParallel(
                self.null_predictor, device_ids=[self.device], **self.kwargs)

    def get_mask(self, embeddings):
        embeddings = embeddings.to(self.args.device)
        out = self.masking_layer(embeddings)
        mask = torch.sigmoid(out)
        return mask

    def get_mask_grad(self):
        weight_grads = []
        bias_grads = []
        for idx in range(self.args.n_layers):
            if isinstance(self.masking_layer[idx], torch.nn.Linear):
                weight_grads.append(self.masking_layer[idx].weight.grad)
                bias_grads.append(self.masking_layer[idx].bias.grad)
        return weight_grads, bias_grads


    def get_checkpoint_dict(self):
        ckpt_dict = self.filter_obj.get_checkpoint_dict()
        if self.args.train_mask:
            ckpt_dict.update(
                {"masking_layer": self.masking_layer.state_dict()})
        return ckpt_dict

    def load_checkpoint(self, checkpoint):
        
        self.masking_layer.load_state_dict(checkpoint["masking_layer"])
        with torch.no_grad():
            if self.args.use_recap_siglip:
                self.logit_scale.copy_(checkpoint["logit_scale"])
                self.logit_bias.copy_(checkpoint["logit_bias"])
        print("Loaded masking layer from checkpoint")


    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):

        '''
        img_embeddings: (batch, dimesnion), text_embeddings: (batch, dimension)
        returns : positive img embeddings conditioned on correpoding positive caption
        
        '''

        info = self.filter_obj.get_latents(img_embeddings)
        sae_outputs_dict = {"info": info, "reconstruction": None}
        
        
        latents, inds, vals = self.filter_obj.unpack_info(info)

        mask = self.get_mask(text_embeddings)

        if "inverted_mask" in kwargs and kwargs["inverted_mask"]:
            mask = 1 - mask

            
        if inds is not None:
            topk_mask = torch.full((mask.shape[0], mask.shape[1]), -1.0, dtype=mask.dtype).to(self.args.device)
            row_indices = torch.arange(inds.size(0)).to(inds.device)
            row_indices = row_indices.unsqueeze(1).expand(-1, inds.size(1))
            partial_mask = mask[row_indices, inds]
            topk_mask[row_indices, inds] = partial_mask

                # take mean
            
            
            if 'which_type' in kwargs and kwargs['which_type'] =='variation1': 
                mask = mask.repeat(text_embeddings.shape[0], 1)
                inds = inds.repeat_interleave(text_embeddings.shape[0], dim=0)
                vals = vals.repeat_interleave(text_embeddings.shape[0], dim=0)
                row_indices = torch.arange(inds.size(0)).to(inds.device)
                row_indices = row_indices.unsqueeze(1).expand(-1, inds.size(1))
                partial_mask = mask[row_indices, inds]

            after_mask_vals = vals * partial_mask
           

            info_updated = {"inds": inds, "vals": after_mask_vals,
                            "latents_pre_act": latents, "mask": mask}
            sae_outputs_dict["info_updated"] = info_updated
            # thsi recon_features are each image conditioned on all texts (i1|t1), (i1|t2), ......, (i1|tN), (i2|t1), (i2|t2), ......, (i2|tN),.....
            # all conditioning of first image comes first
            part_recon_image_features_unnorm = self.filter_obj.get_reconstructed_features(
                img_embeddings, info_updated)
            sae_outputs_dict.update({'topk_mask': topk_mask})


        else:  # No SAE: mask is applied directly on the CLIP embeddings (IdentityMaskable)
            # latents are the CLIP image embeddings (batch, embed_dim); mask is
            # conditioned on the text embeddings (batch, embed_dim).
            topk_mask = mask
            if 'which_type' in kwargs and kwargs['which_type'] == 'variation1':
                # Build all (image_i | text_j) pairs, image slow / text fast, to match the
                # SAE branch ordering: (i1|t1), (i1|t2), ..., (i1|tN), (i2|t1), ...
                latents = latents.repeat_interleave(text_embeddings.shape[0], dim=0)
                mask = mask.repeat(text_embeddings.shape[0], 1)
            after_mask_latents = latents * mask
            info_updated = {"inds": None, "vals": None,
                            "latents": after_mask_latents, "mask": mask}
            sae_outputs_dict["info_updated"] = info_updated
            part_recon_image_features_unnorm = self.filter_obj.get_reconstructed_features(
                img_embeddings, info_updated)
            sae_outputs_dict.update({'topk_mask': topk_mask})


        return part_recon_image_features_unnorm, sae_outputs_dict

    def get_filtered_embeddings_single_image_multiple_texts(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        info = self.filter_obj.get_latents(img_embeddings)
        sae_outputs_dict = {"info": info, "reconstruction": None}
        latents, inds, vals = self.filter_obj.unpack_info(info)

        mask = self.get_mask(text_embeddings)

        if "inverted_mask" in kwargs and kwargs["inverted_mask"]:
            mask = 1 - mask

        if inds is not None:

            after_mask_vals = vals.squeeze() * mask[:, inds.squeeze()]
            if len(inds.shape) == 1:
                inds = inds.unsqueeze(0)
            inds = inds.expand(mask.shape[0], inds.shape[1])
            inds = inds.contiguous()
            info_updated = {"inds": inds, "vals": after_mask_vals,
                            "latents_pre_act": latents, "mask": mask}
            sae_outputs_dict["info_updated"] = info_updated
        else:  # For the identity case where no SAE is there
            after_mask_latents = latents * mask
            info_updated = {"inds": None, "vals": None,
                            "latents": after_mask_latents, "mask": mask}
            sae_outputs_dict["info_updated"] = info_updated


        part_recon_image_features_unnorm = self.filter_obj.get_reconstructed_features(
            img_embeddings, info_updated)

        sae_outputs_dict.update({'topk_mask': mask})


        return part_recon_image_features_unnorm, sae_outputs_dict

    def get_optimizer_param_list(self):
        param_list = self.filter_obj.get_optimizer_param_list()
        if self.args.train_mask:
            param_list += [{'params': list(self.masking_layer.parameters(
            )), 'lr': self.args.train_mask_lr, "fused": True, "name": "filter_algo"}]

        return param_list

    def eval(self):
        self.masking_layer.eval()
        self.filter_obj.eval()

    def train(self):
        self.masking_layer.train()
        self.filter_obj.train()

    def freeze_parameters(self):
        for param in self.masking_layer.parameters():
            param.requires_grad = False
        self.filter_obj.freeze_parameters()

    def unfreeze_parameters(self):
        for param in self.masking_layer.parameters():
            param.requires_grad = True
        self.filter_obj.unfreeze_parameters()


class GTMaskConceptFilter(ConceptFilter):

    def __init__(self, args):
        super().__init__(args)

    def load_checkpoint(self, checkpoint):
        return

    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        info = self.filter_obj.get_latents(
            img_embeddings, custom_inds=kwargs["custom_inds"])
        sae_outputs_dict = {"info": info, "reconstruction": None}

        latents, inds, vals = self.filter_obj.unpack_info(info)

        mask = kwargs["text_mask"]
        row_indices = torch.arange(inds.size(0)).to(inds.device)
        row_indices = row_indices.unsqueeze(1).expand(-1, inds.size(1))
        after_mask_vals = vals * mask[row_indices, inds]

        info_updated = {"inds": inds, "vals": after_mask_vals,
                        "latents_pre_act": latents, "mask": mask}
        part_recon_image_features_unnorm = self.filter_obj.get_reconstructed_features(
            img_embeddings, info_updated)
        sae_outputs_dict["info_updated"] = info_updated

        return part_recon_image_features_unnorm, sae_outputs_dict

    def get_optimizer_param_list(self):
        param_list = self.filter_obj.get_optimizer_param_list()
        return param_list


class CrossAttention(ConceptFilter):
    def __init__(self, args, device, **kwargs):
        super().__init__(args, device)
        self.hidden_size = kwargs['hidden_size']
        self.embed_dim = kwargs['embed_dim']
        self.num_heads = kwargs['num_heads']

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,  # Can adjust number of attention heads
            batch_first=True
        ).to(self.args.device)


        self.img_scale = self.hidden_size ** -0.5
        self.text_scale = self.embed_dim ** -0.5

        self.img_projection_layer = nn.Parameter(self.img_scale * torch.randn(self.hidden_size, self.embed_dim).to(self.args.device))
        self.text_projection_layer = nn.Parameter(self.text_scale * torch.randn(self.embed_dim, self.embed_dim).to(self.args.device))


        self.null_vector = torch.nn.Parameter(
            torch.zeros(output_dim).to(self.args.device))
        torch.nn.init.xavier_normal_(self.null_vector.unsqueeze(0))
        self.null_vector.requires_grad = True


    def init_distributed(self):
        self.filter_obj.init_distributed()
        if self.args.distributed and not self.args.horovod:
            raise NotImplementedError

    def get_mask(self, embeddings):
        raise NotImplementedError

    def get_mask_grads(self):
        raise NotImplementedError


    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):

        unpooled_image_embeddings = kwargs.get("unpooled_image_embeddings", None)
        unpooled_text_embeddings = kwargs.get("unpooled_text_embeddings", None)

        if unpooled_image_embeddings.ndim == 2:
            unpooled_image_embeddings = unpooled_image_embeddings.unsqueeze(0)
        if unpooled_text_embeddings.ndim == 2:
            unpooled_text_embeddings = unpooled_text_embeddings.unsqueeze(0)


        text_embeddings_to_use = unpooled_text_embeddings

        assert (unpooled_image_embeddings.ndim == 3), "Dimension has to be 3"
        projected_image_embeddings = unpooled_image_embeddings @ self.img_projection_layer
        projected_text_embeddings = text_embeddings_to_use @ self.text_projection_layer


        query = projected_text_embeddings

        filtered_embeddings, _ = self.cross_attention(
            query=query,
            key=projected_image_embeddings,
            value=projected_image_embeddings,
            need_weights=False,
        )

        filtered_embeddings = torch.mean(filtered_embeddings, dim=1)


        assert (self.args.train_mask_batch_size != 1 and self.args.batch_size != 1)
        filtered_embeddings = filtered_embeddings.squeeze()


        return filtered_embeddings, {}

    def get_filtered_embeddings_single_image_multiple_texts(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        unpooled_image_embeddings = kwargs.get("unpooled_image_embeddings", None)
        assert unpooled_image_embeddings.ndim == 3
        unpooled_image_embeddings = unpooled_image_embeddings.repeat(text_embeddings.shape[0], 1, 1)
        kwargs["unpooled_image_embeddings"] = unpooled_image_embeddings
        return self.get_filtered_embeddings(imgs, texts, img_embeddings, text_embeddings, **kwargs)

    def get_optimizer_param_list(self):
        param_list = self.filter_obj.get_optimizer_param_list()
        if self.args.train_mask:
            # {'params': list(self.masking_layer.parameters())
            param_list += [{'params': [self.img_projection_layer] + [self.text_projection_layer] + list(self.cross_attention.parameters()), 'lr': self.args.train_mask_lr, "fused": True, "name": "filter_algo"}]
            param_list += [{'params': [self.null_vector] + [self.masking_layer], 'lr': self.args.train_mask_lr, "fused": True, "name": "filter_algo_null_vector"}]
        return param_list

    def load_checkpoint(self, checkpoint):
        with torch.no_grad():
            self.img_projection_layer.copy_(checkpoint["img_projection_layer"])
            self.text_projection_layer.copy_(checkpoint["text_projection_layer"])

        self.cross_attention.load_state_dict(checkpoint["cross_attention"])
        with torch.no_grad():
            self.null_vector.copy_(checkpoint["null_vector"])
            self.masking_layer.copy_(checkpoint["masking_layer"])

    def get_checkpoint_dict(self):
        ckpt_dict = self.filter_obj.get_checkpoint_dict()
        if self.args.train_mask:
            ckpt_dict.update(
                {"img_projection_layer": self.img_projection_layer})
            ckpt_dict.update(
                {"text_projection_layer": self.text_projection_layer})
            ckpt_dict.update(
                {"cross_attention": self.cross_attention.state_dict()})
            ckpt_dict.update({"null_vector": self.null_vector})
            ckpt_dict.update({"masking_layer": self.masking_layer})
        return ckpt_dict

    def eval(self):
        self.cross_attention.eval()
        self.filter_obj.eval()

    def train(self):
        self.cross_attention.train()
        self.filter_obj.train()

    def freeze_parameters(self):
        self.img_projection_layer.requires_grad = False
        self.text_projection_layer.requires_grad = False
        for param in self.cross_attention.parameters():
            param.requires_grad = False
        self.null_vector.requires_grad = False
        self.masking_layer.requires_grad = False
        self.filter_obj.freeze_parameters()


class SimpleMLP(ConceptFilter):

    def __init__(self, args, device, **kwargs):
        super().__init__(args, device, **kwargs)
        self.embed_dim = kwargs.get("embed_dim", None)
        self.logit_scale = torch.nn.Parameter(torch.ones([]).to(device) * np.log(1 / 0.07), requires_grad=True)
        self.logit_bias = torch.nn.Parameter(torch.ones([]).to(device) * -10, requires_grad=True)

    # https://stackoverflow.com/a/49433937/
    def init_weights(self, m, bias_fill=0.0):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:  # Ensure bias exists
                m.bias.data.fill_(bias_fill)


    def set_filtering_architecture(self, filter_obj):
        super().set_filtering_architecture(filter_obj)
        output_dim = self.embed_dim
        self.null_vector = torch.nn.Parameter(
            torch.zeros(output_dim).to(self.args.device))
        torch.nn.init.xavier_normal_(self.null_vector.unsqueeze(0))
        self.null_vector.requires_grad = True

        mask_layer_dims = [2 * self.embed_dim, output_dim]

        layer_list = [torch.nn.Linear(
            mask_layer_dims[0], mask_layer_dims[1])]
        for idx in range(len(mask_layer_dims) - 2):
            layer_list.extend([
                torch.nn.ReLU(),
                torch.nn.Linear(
                    mask_layer_dims[idx + 1], mask_layer_dims[idx + 2])
            ])


        self.masking_layer = torch.nn.Sequential(
            *layer_list).to(self.args.device)

        self.masking_layer.apply(functools.partial(
            self.init_weights, bias_fill=self.args.sigmoid_bias_fill))

    def init_distributed(self):
        self.filter_obj.init_distributed()
        if self.args.distributed and not self.args.horovod:
            raise NotImplementedError("Not implemented for SimpleMLP null vector")
            self.masking_layer = torch.nn.parallel.DistributedDataParallel(
                self.masking_layer, device_ids=[self.device], **self.kwargs)

    def get_mask(self, embeddings):
        raise NotImplementedError

    def get_mask_grads(self):
        raise NotImplementedError

    def get_filtered_embeddings(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        img_embeddings_norm = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        concat_embeddings = torch.cat(
            (img_embeddings_norm, text_embeddings_norm), dim=-1)
        filtered_embeddings = self.masking_layer(concat_embeddings)

        return filtered_embeddings, {}

    def get_filtered_embeddings_single_image_multiple_texts(self, imgs, texts, img_embeddings, text_embeddings, **kwargs):
        img_embeddings = img_embeddings.repeat(text_embeddings.shape[0], 1)
        return self.get_filtered_embeddings(imgs, texts, img_embeddings, text_embeddings, **kwargs)

    def get_optimizer_param_list(self):
        param_list = self.filter_obj.get_optimizer_param_list()
        if self.args.train_mask:
            param_list += [{'params': list(self.masking_layer.parameters(
            )), 'lr': self.args.train_mask_lr, "fused": True, "name": "filter_algo"}]
            param_list += [{'params': [self.null_vector], 'lr': self.args.train_mask_lr, "fused": True, "name": "filter_algo_null_vector"}]
        return param_list

    def load_checkpoint(self, checkpoint):
        self.masking_layer.load_state_dict(checkpoint["masking_layer"])
        with torch.no_grad():
            self.null_vector.copy_(checkpoint["null_vector"])

    def get_checkpoint_dict(self):
        ckpt_dict = self.filter_obj.get_checkpoint_dict()
        if self.args.train_mask:
            ckpt_dict.update(
                {"masking_layer": self.masking_layer.state_dict()})
            ckpt_dict.update({"null_vector": self.null_vector})
        return ckpt_dict

    def eval(self):
        self.masking_layer.eval()
        self.filter_obj.eval()

    def train(self):
        self.masking_layer.train()
        self.filter_obj.train()

    def freeze_parameters(self):
        for param in self.masking_layer.parameters():
            param.requires_grad = False
        self.null_vector.requires_grad = False
        self.filter_obj.freeze_parameters()


# contrastive negative conditioned loss


#         '''
#         img_embeddings: (batch, dimesnion), text_embeddings: (batch, dimension)

        
#         '''

        
#         # if latents is None:
#         #     latents = torch.zeros_like(shape=(self.filter_obj.latent_dim,))
#         #     inds = info['inds']
#         #     row_indices = torch.arange(inds.size(0)).to(inds.device)
#         #     row_indices = row_indices.unsqueeze(1).expand(-1, inds.size(1))
#         #     latents[row_indices, inds] = vals


#         # Binarize the mask to 0 and 1
#         # binary_mask = (mask>0.5).int()
#         # filtered_mask = mask * binary_mask


#             # after_mask_latents = latents * mask
#             # after_mask_vals = after_mask_latents[row_indices,inds]
#                             "latents_pre_act": latents, "mask": mask}
#             # thsi recon_features are each image conditioned on all texts (i1|t1), (i1|t2), ......, (i1|tN), (i2|t1), (i2|t2), ......, (i2|tN),.....
#             # all conditioning of first image comes first
#                 img_embeddings, info_updated)


#         else:  # For the identity case where no SAE is there
            
