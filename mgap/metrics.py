import torch
import torchmetrics
from mgap import config
import numpy as np


class ModalityGapLiang(torchmetrics.Metric):
    """
    Modality Gap metric by Liang et al.
    """

    def __init__(self, embed_dim, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.embed_dim = embed_dim
        self.add_state("embedding_difference", default=torch.zeros(
            embed_dim), dist_reduce_fx=None)
        self.add_state("positive_cossim", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("negative_cossim", default=torch.tensor(0.0), dist_reduce_fx=None)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx=None)
        self.add_state("off_diag_count", default=torch.tensor(0), dist_reduce_fx=None)


    def update(self, img_features_norm, txt_features_norm):
        assert img_features_norm.size(1) == self.embed_dim
        assert txt_features_norm.size(1) == self.embed_dim
        self.embedding_difference += (img_features_norm -
                                      txt_features_norm).sum(dim=0).detach()
    
        # dim=1 as first dimension is number of sample and 2nd dim is embed_dim
        cossim_matrix = img_features_norm @ txt_features_norm.T
        cossim_matrix = cossim_matrix.detach().cpu().numpy()
        
        pos_cossim_single = np.diag(cossim_matrix).sum()
        self.positive_cossim += pos_cossim_single
        negative_cossim_single = cossim_matrix.sum() - pos_cossim_single
        self.negative_cossim += negative_cossim_single

        self.count += img_features_norm.size(0)
        self.off_diag_count += img_features_norm.size(0) * (img_features_norm.size(0) - 1)


    def compute(self):
        mean_embed = self.embedding_difference / self.count
        return torch.linalg.vector_norm(mean_embed, ord=2), mean_embed

    def get_logging_dict(self, panel_name, prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        mgap_val, _ = self.compute()
        mean_positive_cossim = self.positive_cossim/self.count
        mean_negative_cossim = self.negative_cossim/self.off_diag_count
        metrics = {f"{panel_name}/{prefix}neg_cossim": mean_negative_cossim.item()}
        metrics = {f"{panel_name}/{prefix}pos_cossim": mean_positive_cossim.item()}
        metrics.update({f"{panel_name}/{prefix}mgap": mgap_val.item()})
        return metrics


class RMGSchrodi(torchmetrics.Metric):

    def __init__(self, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("rmg", default=[], dist_reduce_fx=None)
        self.add_state("img_intramodal_batch", default=[], dist_reduce_fx=None)
        self.add_state("txt_intramodal_batch", default=[], dist_reduce_fx=None)
        self.add_state("img_txt_intermodal_batch", default=[], dist_reduce_fx=None)

    def update(self, img_features, txt_features, normalized = True):
        # update has to be called once with all features
        if not normalized:
            img_features_norm = img_features / img_features.norm(dim=1, keepdim=True)
            txt_features_norm = txt_features / txt_features.norm(dim=1, keepdim=True)
        else:
            img_features_norm = img_features
            txt_features_norm = txt_features
        
        N = img_features_norm.size(0)
        with torch.no_grad():
            #  use 1- cossim, as it needs distance
            img_txt_cossims = (1 - (img_features_norm @ txt_features_norm.T)).diag().sum().item()
            img_img_cossims_mat = (1 - (img_features_norm @ img_features_norm.T))
            txt_txt_cossims_mat = (1 - (txt_features_norm @ txt_features_norm.T))
            img_img_cossims = img_img_cossims_mat.sum().item() - img_img_cossims_mat.diag().sum().item()
            txt_txt_cossims = txt_txt_cossims_mat.sum().item() - txt_txt_cossims_mat.diag().sum().item()

            
            off_diag_count = N * (N - 1)

            mean_img_img_cossims = img_img_cossims / off_diag_count
            mean_txt_txt_cossims = txt_txt_cossims / off_diag_count
            mean_img_txt_cossims = img_txt_cossims / N
            mean_of_mean_intra_modal = ( mean_img_img_cossims + mean_txt_txt_cossims ) / 2

            numerator = mean_img_txt_cossims
            denominator = mean_of_mean_intra_modal + numerator
            rmg = numerator / denominator
        
        self.img_intramodal_batch.append(mean_img_img_cossims)
        self.txt_intramodal_batch.append(mean_txt_txt_cossims)
        self.rmg = rmg
        self.img_intramodal = mean_img_img_cossims
        self.txt_intramodal = mean_txt_txt_cossims
        self.img_txt_intermodal = mean_img_txt_cossims


    def compute(self):
        return self.rmg

    def get_logging_dict(self, panel_name, prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        # compute removed as now it is done a in single update call using a subset of data points
        rmg_val = self.compute()
        metrics = {f"{panel_name}/{prefix}rmg": rmg_val}
        metrics.update({f"{panel_name}/{prefix}img_intramodal": self.img_intramodal})
        metrics.update({f"{panel_name}/{prefix}txt_intramodal": self.txt_intramodal})
        metrics.update({f"{panel_name}/{prefix}img_txt_intermodal": self.img_txt_intermodal})

        return metrics


class CaptionCosSims(torchmetrics.Metric):
    """
    Get cosine similarities between positive pairs, negative pairs, and all pairs
    """

    def __init__(self, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("pos_cossims", default=[], dist_reduce_fx=None)
        self.add_state("neg_cossims", default=[], dist_reduce_fx=None)
        self.add_state("all_cossims", default=[], dist_reduce_fx=None)
        self.add_state("pos_count", default=torch.tensor(0),
                       dist_reduce_fx=None)
        self.add_state("neg_count", default=torch.tensor(0),
                       dist_reduce_fx=None)
        self.add_state("all_count", default=torch.tensor(0),
                       dist_reduce_fx=None)

    def update(self, cossims):
        pos_cossims = (cossims).diagonal().tolist()
        neg_cossims = self._get_off_diagonal_elements(
            cossims).flatten().tolist()
        all_cossims = cossims.flatten().tolist()
        self.pos_cossims += pos_cossims
        self.all_cossims += all_cossims
        self.neg_cossims += neg_cossims
        self.pos_count += len(pos_cossims)
        self.neg_count += len(neg_cossims)
        self.all_count += len(all_cossims)

    def compute(self):
        mean_pos_cossim = sum(self.pos_cossims) / self.pos_count
        mean_neg_cossim = sum(self.neg_cossims) / self.neg_count
        mean_all_cossim = sum(self.all_cossims) / self.all_count
        return {
            "pos_cossims": self.pos_cossims,
            "neg_cossims": self.neg_cossims,
            "all_cossims": self.all_cossims,
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "all_count": self.all_count,
            "mean_pos_cossim": mean_pos_cossim,
            "mean_neg_cossim": mean_neg_cossim,
            "mean_all_cossim": mean_all_cossim,
        }

    def get_logging_dict(self, split, prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        caption_cossims_dict = self.compute()
        cos_sim_metrics = {f"{split}_{prefix}caption_cossims_mean_pos": caption_cossims_dict['mean_pos_cossim'].item(
        ), f"{split}_{prefix}caption_cossims_mean_neg": caption_cossims_dict['mean_neg_cossim'].item(), f"{split}_{prefix}caption_cossims_mean_all": caption_cossims_dict['mean_all_cossim'].item()}
        return cos_sim_metrics

    # https://discuss.pytorch.org/t/most-efficient-way-to-get-just-the-off-diagonal-elements-of-a-tensor/131065/2
    def _get_off_diagonal_elements(self, M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)]


class CaptionCosSimsSimple(torchmetrics.Metric):
    """
    Get cosine similarities between positive pairs, negative pairs, and all pairs
    """

    def __init__(self, dist_sync_on_step=False, process_group=None, return_pos_cossims = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("pos_cossims", default=torch.tensor(
            0.0), dist_reduce_fx=None)
        self.add_state("neg_cossims", default=torch.tensor(
            0.0), dist_reduce_fx=None)
        self.add_state("all_cossims", default=torch.tensor(
            0.0), dist_reduce_fx=None)
        self.add_state("pos_count", default=torch.tensor(0), dist_reduce_fx=None)
        self.add_state("neg_count", default=torch.tensor(0), dist_reduce_fx=None)
        self.add_state("all_count", default=torch.tensor(0), dist_reduce_fx=None)

        # For img-img
        self.add_state("img_img_cossims", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("img_count", default=torch.tensor(0), dist_reduce_fx=None)

        # For text-text
        self.add_state("text_text_cossims", default=torch.tensor(0.0), dist_reduce_fx=None)
        self.add_state("text_count", default=torch.tensor(0), dist_reduce_fx=None)
        self.return_pos_cossims = return_pos_cossims
        if self.return_pos_cossims:
            self.add_state("all_pos_cossims", default=[], dist_reduce_fx=None) 
        else: 
            self.all_pos_cossims = None

    def update(self, cossims, img_img_cossims=None, text_text_cossims=None):

        pos_cossims = (cossims).diagonal()
        if self.return_pos_cossims:
            self.all_pos_cossims.append(pos_cossims.cpu())
        
        neg_cossims = self._get_off_diagonal_elements(cossims).flatten()
        all_cossims = cossims.flatten()
        self.pos_cossims += pos_cossims.sum().item()
        self.all_cossims += all_cossims.sum().item()
        self.neg_cossims += neg_cossims.sum().item()
        self.pos_count += len(pos_cossims)
        self.neg_count += len(neg_cossims)
        self.all_count += len(all_cossims)
        if img_img_cossims is not None:
            off_diagonal_elements = self._get_off_diagonal_elements(img_img_cossims).flatten()
            self.img_img_cossims += off_diagonal_elements.sum().item()
            self.img_count += len(off_diagonal_elements)
        if text_text_cossims is not None:
            off_diagonal_elements = self._get_off_diagonal_elements(text_text_cossims).flatten()
            self.text_text_cossims += off_diagonal_elements.sum().item()
            self.text_count += len(off_diagonal_elements)

    def compute(self):
        mean_pos_cossim = self.pos_cossims / self.pos_count
        mean_neg_cossim = self.neg_cossims / self.neg_count
        mean_all_cossim = self.all_cossims / self.all_count
        if self.return_pos_cossims:
            self.all_pos_cossims = torch.cat(self.all_pos_cossims).cpu()

        dict = {
            "sum_pos_cossims": self.pos_cossims,
            "sum_neg_cossims": self.neg_cossims,
            "sum_all_cossims": self.all_cossims,
            "pos_count": self.pos_count,
            "neg_count": self.neg_count,
            "all_count": self.all_count,
            "mean_pos_cossim": mean_pos_cossim,
            "mean_neg_cossim": mean_neg_cossim,
            "mean_all_cossim": mean_all_cossim,
        }
        if self.img_count > 0:
            mean_img_img_cossim = self.img_img_cossims / self.img_count
            dict.update({'mean_img_img_cossim': mean_img_img_cossim,
                        "img_count": self.img_count, "sum_img_img_cossims": self.img_img_cossims})
        if self.text_count > 0:
            mean_text_text_cossim = self.text_text_cossims / self.text_count
            dict.update({'mean_text_text_cossim': mean_text_text_cossim,
                        "text_count": self.text_count, "sum_text_text_cossims": self.text_text_cossims})

       
        return dict

    def get_logging_dict(self, panel_name="", prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        caption_cossims_dict = self.compute()
        cos_sim_metrics = {f"{panel_name}/{prefix}img_text_cossims_mean_pos": caption_cossims_dict['mean_pos_cossim'].item(), f"{panel_name}/{prefix}img_text_cossims_mean_neg": caption_cossims_dict['mean_neg_cossim'].item(), f"{panel_name}/{prefix}img_text_cossims_mean_all": caption_cossims_dict['mean_all_cossim'].item()}
        if 'mean_img_img_cossim' in caption_cossims_dict:
            cos_sim_metrics.update({f"{panel_name}/{prefix}img_img_cossims_mean_all": caption_cossims_dict['mean_img_img_cossim'].item()})
        if 'mean_text_text_cossim' in caption_cossims_dict:
            cos_sim_metrics.update({f"{panel_name}/{prefix}text_text_cossims_mean_all": caption_cossims_dict['mean_text_text_cossim'].item()})
        
        return cos_sim_metrics, self.all_pos_cossims
       

    # https://discuss.pytorch.org/t/most-efficient-way-to-get-just-the-off-diagonal-elements-of-a-tensor/131065/2
    def _get_off_diagonal_elements(self, M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)]


class ContrastiveLossNumeratorDenominator(torchmetrics.Metric):
    """
    Get numerator and denominator from contrastive loss
    """

    def __init__(self, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.add_state("numerator", default=[], dist_reduce_fx=None)
        self.add_state("denominator_row", default=[], dist_reduce_fx=None)
        self.add_state("denominator_col", default=[], dist_reduce_fx=None)
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, cossims):
        assert cossims.size(0) == cossims.size(1)
        exp_cossims = torch.exp(cossims)
        self.numerator += exp_cossims.diagonal().tolist()
        self.denominator_row += exp_cossims.sum(dim=1).tolist()
        self.denominator_col += exp_cossims.sum(dim=0).tolist()
        self.count += cossims.size(0)

    def compute(self):
        assert len(self.numerator) == len(self.denominator_row)
        assert len(self.numerator) == len(self.denominator_col)
        mean_numerator = sum(self.numerator) / self.count
        mean_denominator_row = sum(self.denominator_row) / self.count
        mean_denominator_col = sum(self.denominator_col) / self.count
        return {
            "numerator": self.numerator,
            "denominator_row": self.denominator_row,
            "denominator_col": self.denominator_col,
            "count": self.count,
            "mean_numerator": mean_numerator,
            "mean_denominator_row": mean_denominator_row,
            "mean_denominator_col": mean_denominator_col
        }

    def get_logging_dict(self, split, prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        contrastive_loss_dict = self.compute()
        contrastive_loss_metrics = {f"{split}_{prefix}contrastive_loss_mean_numerator": contrastive_loss_dict['mean_numerator'].item(),
                                    f"{split}_{prefix}contrastive_loss_mean_denom_row": contrastive_loss_dict['mean_denominator_row'].item(),
                                    f"{split}_{prefix}contrastive_loss_mean_denom_col": contrastive_loss_dict['mean_denominator_col'].item()}
        return contrastive_loss_metrics


class MADZeroShot(torchmetrics.Metric):
    """
    Get zero shot accuracy for MAD
    """

    def __init__(self, attribute_name, zeroshot_weight, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.attribute_name = attribute_name
        self.zeroshot_weight = zeroshot_weight
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx=None)
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, img_features, labels, use_extra_ortho_dim=False):
        img_features_norm = img_features / \
            img_features.norm(dim=1, keepdim=True)
        if use_extra_ortho_dim:
            img_features_norm = img_features_norm[:, 0:-1]
        cossims = img_features_norm @ self.zeroshot_weight.T
        preds = cossims.argmax(dim=1)
        self.correct += (preds == labels).sum().item()
        self.count += img_features_norm.size(0)

        preds_result = torch.tensor(config.mad_attribute_pos_dict[self.attribute_name])[preds.cpu()]
        return preds_result

    def compute(self):
        accuracy = (self.correct / self.count).item()
        return {
            "attribute_name": self.attribute_name,
            "correct": self.correct,
            "count": self.count,
            "accuracy": accuracy,
        }


class MADMaskPredictionAccuracy(torchmetrics.Metric):
    """
    Get accuracy of predicted MAD masks from the learnt mask module
    """

    def __init__(self, threshold=0.5, dist_sync_on_step=False, process_group=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.threshold = threshold
        self.add_state("all_correct", default=torch.tensor(0),
                       dist_reduce_fx=None)
        self.add_state("all_count", default=torch.tensor(0),
                       dist_reduce_fx=None)

    def update(self, mask_preds, mask_gt):
        assert mask_preds.size() == mask_gt.size()
        hard_mask_preds = mask_preds > self.threshold
        self.all_correct += (hard_mask_preds == mask_gt).sum().item()
        self.all_count += mask_gt.size(0) * mask_gt.size(1)

    def compute(self):
        accuracy_dict = {}
        all_accuracy = (self.all_correct / self.all_count).item()
        accuracy_dict["all"] = all_accuracy
        return accuracy_dict

    def get_logging_dict(self, split, prefix=""):
        if prefix != "":
            prefix = f"{prefix}_"
        mask_accuracy_dict = self.compute()
        mask_accuracy_metrics = {f"{split}_{prefix}mask_accuracy_{attr_name}":
                                 mask_accuracy_dict[attr_name] for attr_name in mask_accuracy_dict}
        return mask_accuracy_metrics


class MADMaskPredictionTextAccuracy(MADMaskPredictionAccuracy):

    def __init__(self, threshold=0.5, dist_sync_on_step=False, process_group=None):
        super().__init__(threshold=threshold,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group)
        self.cumulative_26_to_6_indices = self.get_cumulatiave_26_to_6_indices()
        self.add_state("gt_one_counts", default=[], dist_reduce_fx=None)
        self.add_state("pred_one_counts", default=[], dist_reduce_fx=None)
        self.add_state("total_selected_counts",
                       default=[], dist_reduce_fx=None)

    def get_mask_gt_from_gt_attrs(self, pred_mask, img_mask, text_mask, inds):
        row_indices = torch.arange(inds.size(0)).to(inds.device)
        row_indices = row_indices.unsqueeze(1).expand(-1, inds.size(1))
        text_mask_6 = self.convert_26_to_6(text_mask)
        mask_gt_full = []
        for idx, (k, v) in enumerate(config.mad_attribute_pos_dict.items()):
            mask_gt_full.append(
                text_mask_6[:, idx].unsqueeze(1).expand(-1, len(v)))
        mask_gt_full = torch.cat(mask_gt_full, dim=1).to(pred_mask.device)
        mask_gt_selected = mask_gt_full[row_indices, inds]
        pred_mask_selected = pred_mask[row_indices, inds]
        return pred_mask_selected, mask_gt_selected

    def convert_26_to_6(self, mask):
        mask_6 = torch.zeros(mask.size(0), 6).to(mask.device)
        for i, (start, end) in enumerate(zip(self.cumulative_26_to_6_indices[:-1], self.cumulative_26_to_6_indices[1:])):
            mask_6[:, i] = mask[:, start:end].sum(dim=1)
        assert mask_6.max() == 1
        return mask_6

    def get_cumulatiave_26_to_6_indices(self):
        cumulative_26_to_6_indices = [0]
        for k, v in config.mad_attribute_pos_dict.items():
            cumulative_26_to_6_indices.append(
                cumulative_26_to_6_indices[-1] + len(v))
        return cumulative_26_to_6_indices

    def update(self, pred_mask, img_mask, text_mask, inds):
        pred_mask_selected, mask_gt_selected = self.get_mask_gt_from_gt_attrs(
            pred_mask, img_mask, text_mask, inds)
        super().update(pred_mask_selected, mask_gt_selected)
        self.gt_one_counts += mask_gt_selected.sum(dim=1).tolist()
        self.pred_one_counts += (pred_mask_selected >
                                 self.threshold).sum(dim=1).tolist()
        self.total_selected_counts += [
            mask_gt_selected.size(1)] * mask_gt_selected.size(0)

    def compute(self):
        # WARNING: does not compute for parent class! Don't use directly
        parent_output_dict = super().compute()
        self.gt_one_counts = torch.tensor(self.gt_one_counts).float()
        self.pred_one_counts = torch.tensor(self.pred_one_counts).float()
        self.total_selected_counts = torch.tensor(
            self.total_selected_counts).float()
        child_output_dict = {
            "gt_one_counts": {
                "mean": self.gt_one_counts.mean().item(),
                "std": self.gt_one_counts.std().item(),
                "min": self.gt_one_counts.min().item(),
                "max": self.gt_one_counts.max().item(),
            },
            "pred_one_counts": {
                "mean": self.pred_one_counts.mean().item(),
                "std": self.pred_one_counts.std().item(),
                "min": self.pred_one_counts.min().item(),
                "max": self.pred_one_counts.max().item(),
            },
            "total_selected_counts": {
                "mean": self.total_selected_counts.mean().item(),
                "std": self.total_selected_counts.std().item(),
                "min": self.total_selected_counts.min().item(),
                "max": self.total_selected_counts.max().item(),
            }
        }
        return {**parent_output_dict, **child_output_dict}

    def get_logging_dict(self, split, prefix=""):
        mask_text_accuracy_dict = self.compute()
        mask_text_accuracy_out_dict = {}
        for k, v in mask_text_accuracy_dict.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    mask_text_accuracy_out_dict[f"{split}_{prefix}{k}.{k1}"] = v1
            else:
                mask_text_accuracy_out_dict[f"{split}_{prefix}{k}"] = v
        return mask_text_accuracy_out_dict

    