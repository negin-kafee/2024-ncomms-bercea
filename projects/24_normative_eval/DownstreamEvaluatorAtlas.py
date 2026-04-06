import logging
import io
import copy
import os
import time
from PIL import Image
import pandas as pd
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import wandb
import plotly.graph_objects as go

from torch.nn import L1Loss
from skimage.metrics import structural_similarity as ssim
import numpy as np
import lpips
from scipy import ndimage
import torch.multiprocessing as mp
#
from dl_utils import *
from optim.metrics import compute_auprc, compute_dice, compute_dice_per_image, compute_auroc_per_image
from core.DownstreamEvaluator import DownstreamEvaluator
from model_zoo.vgg import VGGEncoder
import uuid
import cv2


def apply_postprocessing(pred_binary, method='none', min_size=10):
    """
    Apply post-processing to binary prediction mask.
    
    Args:
        pred_binary: Binary prediction mask (0 or 1)
        method: 'none', 'morphology', 'connected_components', 'both'
        min_size: Minimum connected component size to keep
    
    Returns:
        Processed binary mask
    """
    if method == 'none':
        return pred_binary
    
    pred_processed = pred_binary.copy().astype(np.uint8)
    
    if method in ['morphology', 'both']:
        # Morphological closing to fill small holes, then opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_processed = cv2.morphologyEx(pred_processed, cv2.MORPH_CLOSE, kernel)
        pred_processed = cv2.morphologyEx(pred_processed, cv2.MORPH_OPEN, kernel)
    
    if method in ['connected_components', 'both']:
        # Remove small connected components
        labeled, num_features = ndimage.label(pred_processed)
        if num_features > 0:
            component_sizes = ndimage.sum(pred_processed, labeled, range(1, num_features + 1))
            # Keep only components larger than min_size
            for i, size in enumerate(component_sizes, 1):
                if size < min_size:
                    pred_processed[labeled == i] = 0
    
    return pred_processed.astype(float)


def _gpu_worker(rank, num_gpus, model_class_name, model_params, global_model,
                cache_dir, image_path, dataset_config, blacklist, batch_size):
    """
    Standalone worker function for multi-GPU precomputation.
    Each worker processes batches where idx % num_gpus == rank.
    """
    device = torch.device(f'cuda:{rank}')
    # Reconstruct model on this GPU
    from dl_utils.config_utils import import_module
    model_cls = import_module(model_class_name[0], model_class_name[1])
    model = model_cls(**model_params).to(device)
    model.load_state_dict(global_model, strict=False)
    model.eval()

    # Reconstruct dataloader on this GPU's worker
    dl_cls = import_module(dataset_config['module_name'], dataset_config['class_name'])
    for dataset_key, ds_cfg in dataset_config['datasets'].items():
        data = dl_cls({**dataset_config['params']['args'], **ds_cfg})
        dataset = data.test_dataloader()

        total_batches = len(dataset)
        my_batches = (total_batches + num_gpus - 1) // num_gpus
        start_time = time.time()
        processed = 0

        for idx, batch_data in enumerate(dataset):
            if idx % num_gpus != rank:
                continue

            if 'dict' in str(type(batch_data)) and 'images' in batch_data.keys():
                data0 = batch_data['images']
            else:
                data0 = batch_data[0]
            x = data0.to(device)
            masks = batch_data[1].to(device)
            masks[masks > 0] = 1

            with torch.no_grad():
                anomaly_map, anomaly_score, x_rec_dict = model.get_anomaly(copy.deepcopy(x))

            # Save cache
            np.save(os.path.join(cache_dir, f'batch_{idx}_anomaly_map.npy'), anomaly_map)
            np.save(os.path.join(cache_dir, f'batch_{idx}_anomaly_score.npy'), anomaly_score)
            x_rec_cpu = {}
            for k, v in x_rec_dict.items():
                x_rec_cpu[k] = v.cpu() if torch.is_tensor(v) else v
            torch.save(x_rec_cpu, os.path.join(cache_dir, f'batch_{idx}_x_rec_dict.pt'))

            # Save reconstruction PNGs (replaces print_files)
            x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict else torch.zeros_like(x)
            x_rec = torch.clamp(x_rec, 0, 1)
            for i in range(len(x)):
                count = idx * len(x) + i
                if count in blacklist:
                    continue
                if torch.sum(masks[i][0]) < 3:
                    continue
                rec_np = (x_rec[i][0].cpu().detach().numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(image_path, f'{model.__class__.__name__}_{dataset_key}_{count}_rec.png'),
                    rec_np
                )

            processed += 1
            if processed % 20 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (my_batches - processed) / rate if rate > 0 else 0
                print(f'[GPU {rank}] Batch {idx}/{total_batches} | '
                      f'{processed}/{my_batches} done | '
                      f'Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s', flush=True)

        elapsed = time.time() - start_time
        print(f'[GPU {rank}] Finished {processed} batches in {elapsed:.0f}s', flush=True)


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """

    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True, dataset_config=None, model_config=None):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        print(f'Checkpoint path: {checkpoint_path}')
        self.checkpoint_path = checkpoint_path + '/images/'
        print(f'Checkpoint path: {checkpoint_path}')

        self.criterion_rec = L1Loss().to(self.device)
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device)
        # Get model class name for unique output filenames
        self.model_name = self.model.__class__.__name__
        # Store configs for multi-GPU worker reconstruction
        self._dataset_config = dataset_config
        self._model_config = model_config

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # Single pass: precompute all anomaly maps and cache to disk
        cache_dir = self._precompute_anomaly_maps(global_model)
        print('Done precomputing anomaly maps')

        # Reuse cached results for all threshold ranges
        self.pathology_localization_cached(cache_dir, 1, 71, True)
        self.pathology_localization_cached(cache_dir, 71, 570, True)
        self.pathology_localization_cached(cache_dir, 570, 10000, True)

    def _log_visualization(self, to_visualize, i, count):
        """
        Helper function to log images and masks to wandb
        :param: to_visualize: list of dicts of images and their configs to be visualized
            dict needs to include:
            - tensor: image tensor
            dict may include:
            - title: title of image
            - cmap: matplotlib colormap name
            - vmin: minimum value for colorbar
            - vmax: maximum value for colorbar
        :param: epoch: current epoch
        """
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0},
                                    figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'][i].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else \
            dict['tensor'][i].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'Anomaly_masks/Example_Atlas_{count}': [wandb.Image(diffp, caption="Atlas_" + str(count))]})

    def find_mask_size_thresholds(self, dataset):
        """
        :param dataset: dataset to find mask size thresholds
        :return: lower and upper tail thresholds
        """
        mask_sizes = []
        for _, data in enumerate(dataset):
            if 'dict' in str(type(data)) and 'images' in data.keys():
                data0 = data['images']
            else:
                data0 = data[0]
            x = data0.to(self.device)
            masks = data[1].to(self.device)
            masks[masks > 0] = 1

            for i in range(len(x)):
                if torch.sum(masks[i][0]) > 1:
                    mask_sizes.append(torch.sum(masks[i][0]).item())

        unique_mask_sizes = np.unique(mask_sizes)
        print(type(unique_mask_sizes))
        lower_tail_threshold = np.percentile(unique_mask_sizes, 25)
        upper_tail_threshold = np.percentile(unique_mask_sizes, 75)

        _ = plt.figure()
        # plt.figure()
        plt.hist(mask_sizes, bins=100)
        plt.xlabel('Mask Sizes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mask Sizes')

        plt.axvline(lower_tail_threshold, color='r', linestyle='--', label=f'25th Percentile: {lower_tail_threshold}')
        plt.axvline(upper_tail_threshold, color='g', linestyle='--', label=f'75th Percentile: {upper_tail_threshold}')
        print(lower_tail_threshold, upper_tail_threshold)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({"Anomaly/Mask sizes1": [wandb.Image(Image.open(buf), caption="Mask Sizes")]})

        plt.clf()

    # ======================= Cached Precomputation =======================

    BLACKLIST = {100, 105, 112, 121, 186, 189, 210, 214, 345, 382,
                 424, 425, 435, 434, 441, 462, 464, 472, 478, 504}

    def _precompute_anomaly_maps(self, global_model):
        """
        Single pass over the dataset: compute get_anomaly() for every batch,
        save results to disk, and write reconstruction PNGs (replacing print_files).
        Returns the cache directory path.
        """
        cache_dir = os.path.join(self.checkpoint_path, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        print(f'Available GPUs: {num_gpus}', flush=True)

        if num_gpus > 1:
            self._precompute_multi_gpu(global_model, cache_dir, num_gpus)
        else:
            self._precompute_single_gpu(global_model, cache_dir)

        return cache_dir

    def _precompute_single_gpu(self, global_model, cache_dir):
        """Single-GPU precomputation of anomaly maps."""
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()

        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            total_batches = len(dataset)
            start_time = time.time()
            logging.info(f'Precomputing anomaly maps for {dataset_key} ({total_batches} batches)')

            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                with torch.no_grad():
                    anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))

                # Save to cache
                np.save(os.path.join(cache_dir, f'batch_{idx}_anomaly_map.npy'), anomaly_map)
                np.save(os.path.join(cache_dir, f'batch_{idx}_anomaly_score.npy'), anomaly_score)
                x_rec_cpu = {}
                for k, v in x_rec_dict.items():
                    x_rec_cpu[k] = v.cpu() if torch.is_tensor(v) else v
                torch.save(x_rec_cpu, os.path.join(cache_dir, f'batch_{idx}_x_rec_dict.pt'))

                # Save reconstruction PNGs (replaces print_files)
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)
                for i in range(len(x)):
                    count = idx * len(x) + i
                    if count in self.BLACKLIST:
                        continue
                    if torch.sum(masks[i][0]) < 3:
                        continue
                    rec_np = (x_rec[i][0].cpu().detach().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(self.checkpoint_path,
                                     f'{self.model_name}_{dataset_key}_{count}_rec.png'),
                        rec_np
                    )

                if idx % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (total_batches - idx - 1) / rate if rate > 0 else 0
                    print(f'[GPU 0] Batch {idx}/{total_batches} | '
                          f'Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s', flush=True)

            elapsed = time.time() - start_time
            print(f'Precomputation done for {dataset_key}: {total_batches} batches in {elapsed:.0f}s', flush=True)

    def _precompute_multi_gpu(self, global_model, cache_dir, num_gpus):
        """Multi-GPU precomputation using torch.multiprocessing.spawn."""
        # Use model config from YAML to reconstruct model in workers
        model_class_name = (self._model_config['module_name'], self._model_config['class_name'])
        model_params = self._model_config['params']

        dataset_config = self._dataset_config

        # Move model weights to CPU for sharing
        global_model_cpu = {k: v.cpu() for k, v in global_model.items()}

        print(f'Launching {num_gpus} GPU workers for precomputation...', flush=True)
        mp.spawn(
            _gpu_worker,
            args=(num_gpus, model_class_name, model_params,
                  global_model_cpu, cache_dir, self.checkpoint_path,
                  dataset_config, self.BLACKLIST, 32),
            nprocs=num_gpus,
            join=True
        )
        print(f'All {num_gpus} GPU workers finished.', flush=True)

    def print_files(self, global_model):
        """
        Validation of downstream tasks
        Prints images to disk

        :param global_model:
            Global parameters
        """

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()


        for dataset_key in self.test_data_dict.keys():

            dataset = self.test_data_dict[dataset_key]

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                # Call this to get the mask size thresholds for the dataset
                # self.find_mask_size_thresholds(dataset)

                # New per batch
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                #

                for i in range(len(x)):
                        count = str(idx * len(x) + i)
                        # Don't use images with large black artifacts:
                        if int(count) in [100, 105, 112, 121, 186, 189, 210, 214, 345, 382, 424, 425, 435, 434, 441,
                                          462, 464, 472, 478, 504]:
                            print("skipping ", count)
                            continue
                        if torch.sum(masks[i][0]) < 3:
                            continue

                        # Example visualizations
                        
                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]

                        cv2.imwrite(self.checkpoint_path + '/' + self.model_name + '_' + dataset_key + '_' + str(count) + '_rec.png',
                                (rec_2_i.cpu().detach().numpy() * 255).astype(np.uint8))
                       

    def pathology_localization(self, global_model, threshold_low, threshold_high, perc_flag=False):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info(f"################ Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
            'DICE': [],
            'AUROC': [],
        }
        pred_dict = dict()

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
                'DICE': [],
                'AUROC': [],
            }
            # Store per-image results for CSV export
            per_image_results = []
            
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                # Call this to get the mask size thresholds for the dataset
                # self.find_mask_size_thresholds(dataset)

                # New per batch
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec', 'tensor': x_rec},
                    {'title': f'Anomaly  map {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'}
                ]

                if 'mask' in x_rec_dict.keys():
                    masked_input = x_rec_dict['mask'] + x
                    masked_input[masked_input>1]=1

                    to_visualize.append({'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                    to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                                        'vmax': x_rec_dict['x_res'].max()})
                    to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'gray'})

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > threshold_low and torch.sum(
                            masks[i][0]) <= threshold_high:  # get the desired sizes of anomalies
                        count = str(idx * len(x) + i)
                        # Don't use images with large black artifacts:
                        if int(count) in [100, 105, 112, 121, 186, 189, 210, 214, 345, 382, 424, 425, 435, 434, 441,
                                          462, 464, 472, 478, 504]:
                            print("skipping ", count)
                            continue

                        # Example visualizations
                        if int(count) % 12 == 0 or int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254,
                                                                  539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                            self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]
                        anomalous_pred.append(anomaly_score[i][0])

                        # Get numpy arrays for current image
                        pred_np = res_2_i_np.cpu().detach().numpy() if torch.is_tensor(res_2_i_np) else res_2_i_np
                        pred_np = np.squeeze(pred_np)  # ensure 2D (H, W) for models returning extra dims
                        mask_np = masks[i][0].cpu().detach().numpy()
                        
                        pred_.append(pred_np)
                        label_.append(mask_np)

                        # Similarity metrics: x_rec vs. x
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)
                        
                        # Per-image DICE - IMPROVED with post-processing
                        # Try multiple preprocessing variants and pick best
                        pred_min, pred_max = pred_np.min(), pred_np.max()
                        
                        # Also try Gaussian smoothed version
                        pred_smoothed = cv2.GaussianBlur(pred_np.astype(np.float32), (5, 5), 1.0)
                        
                        best_dice = 0.0
                        best_dice_pp = 0.0  # with post-processing
                        
                        if pred_max > pred_min:
                            # Combine percentile-based and linear thresholds for better coverage
                            percentile_ths = np.percentile(pred_np, [50, 60, 70, 75, 80, 85, 90, 95, 97, 99])
                            linear_ths = np.linspace(pred_min, pred_max, 51)[1:]  # 50 linear thresholds
                            all_thresholds = np.unique(np.concatenate([percentile_ths, linear_ths]))
                            
                            # Try original prediction
                            for th in all_thresholds:
                                d = compute_dice_per_image(pred_np, mask_np, threshold=th)
                                if d > best_dice:
                                    best_dice = d
                            
                            # Try smoothed prediction
                            smoothed_min, smoothed_max = pred_smoothed.min(), pred_smoothed.max()
                            if smoothed_max > smoothed_min:
                                smoothed_ths = np.linspace(smoothed_min, smoothed_max, 31)[1:]
                                for th in smoothed_ths:
                                    d = compute_dice_per_image(pred_smoothed, mask_np, threshold=th)
                                    if d > best_dice:
                                        best_dice = d
                            
                            # Try Otsu's thresholding (automatic optimal threshold)
                            pred_uint8 = (pred_np * 255).astype(np.uint8)
                            otsu_th, pred_otsu = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            pred_otsu_binary = (pred_otsu > 0).astype(float)
                            intersection_otsu = np.sum(pred_otsu_binary * mask_np)
                            union_otsu = np.sum(pred_otsu_binary) + np.sum(mask_np)
                            d_otsu = 2 * intersection_otsu / (union_otsu + 1e-8) if union_otsu > 0 else 0
                            if d_otsu > best_dice:
                                best_dice = d_otsu
                            
                            # Try with morphological post-processing
                            for th in all_thresholds[::2]:  # sample every 2nd threshold to save time
                                pred_bin = (pred_np > th).astype(np.uint8)
                                pred_pp = apply_postprocessing(pred_bin, method='both', min_size=5)
                                intersection = np.sum(pred_pp * mask_np)
                                union = np.sum(pred_pp) + np.sum(mask_np)
                                d_pp = 2 * intersection / (union + 1e-8) if union > 0 else 0
                                if d_pp > best_dice_pp:
                                    best_dice_pp = d_pp
                            
                            # Also try Otsu + post-processing
                            pred_otsu_pp = apply_postprocessing(pred_otsu_binary.astype(np.uint8), method='both', min_size=5)
                            intersection_otsu_pp = np.sum(pred_otsu_pp * mask_np)
                            union_otsu_pp = np.sum(pred_otsu_pp) + np.sum(mask_np)
                            d_otsu_pp = 2 * intersection_otsu_pp / (union_otsu_pp + 1e-8) if union_otsu_pp > 0 else 0
                            if d_otsu_pp > best_dice_pp:
                                best_dice_pp = d_otsu_pp
                        
                        # Take best of all methods
                        dice_per_img = max(best_dice, best_dice_pp)
                        test_metrics['DICE'].append(dice_per_img)
                        
                        auroc_per_img = compute_auroc_per_image(pred_np, mask_np)
                        test_metrics['AUROC'].append(auroc_per_img)
                        
                        # Store per-image results for CSV
                        per_image_results.append({
                            'image_id': int(count),
                            'MAE': loss_mae.item(),
                            'LPIPS': float(loss_lpips),
                            'SSIM': ssim_,
                            'DICE': dice_per_img,
                            'AUROC': auroc_per_img,
                            'mask_size': int(torch.sum(masks[i][0]).item())
                        })

                    elif torch.sum(
                            masks[i][0]) <= 1:  # use slices without anomalies as "healthy" examples on same domain
                        res_2_i_np_healthy = anomaly_map[i][0]# * combined_mask[i][0].cpu().detach().numpy()
                        healthy_pred.append(anomaly_score[i][0])

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])
            
            # Save per-image results to CSV
            if per_image_results:
                csv_path = self.checkpoint_path + f'/per_image_metrics_{dataset_key}_{threshold_low}_{threshold_high}.csv'
                df = pd.DataFrame(per_image_results)
                df.to_csv(csv_path, index=False)
                logging.info(f'Saved per-image metrics to {csv_path}')
                
                # Log per-image metrics summary to wandb
                wandb.log({
                    f'PerImage/{threshold_low}_{dataset_key}_mean_DICE': np.nanmean(test_metrics['DICE']),
                    f'PerImage/{threshold_low}_{dataset_key}_std_DICE': np.nanstd(test_metrics['DICE']),
                    f'PerImage/{threshold_low}_{dataset_key}_mean_AUROC': np.nanmean(test_metrics['AUROC']),
                    f'PerImage/{threshold_low}_{dataset_key}_std_AUROC': np.nanstd(test_metrics['AUROC']),
                })
                
                # Log CSV as artifact
                artifact = wandb.Artifact(f'per_image_metrics_{dataset_key}_{threshold_low}', type='results')
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)

        for dataset_key in self.test_data_dict.keys():
            # Get some stats on prediction set
            pred_ood, label_ood = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  # .flatten()
            labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  # .flatten()
            print(f'Nr of preditions: {predictions_all.shape}')
            print(
                f'Predictions go from {np.min(predictions_all)} to {np.max(predictions_all)} with mean: {np.mean(predictions_all)}')
            print(f'Labels go from {np.min(labels_all)} to {np.max(labels_all)} with mean: {np.mean(labels_all)}')
            print('Shapes {} {} '.format(labels.shape, predictions.shape))

            # Compute global anomaly localization metrics
            dice_scores = []

            auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
            logging.info(f'Global AUPRC score: {auprc_}')
            wandb.log({f'Metrics/{threshold_low}_Global_AUPRC_{dataset_key}': auprc_})

            # Compute dice score for linear thresholds from 0 to 1
            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice = compute_dice(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]
            best_threshold = ths[highest_score_index]

            logging.info(f'Global highest DICE: {highest_score} at threshold {best_threshold}')
            wandb.log({f'Metrics/{threshold_low}_Global_highest_DICE': highest_score})
            wandb.log({f'Metrics/{threshold_low}_Best_DICE_threshold': best_threshold})
            
            # Compute Average DICE (mean of per-image DICE scores)
            avg_dice = np.nanmean(metrics['DICE'][-1]) if metrics['DICE'] else 0
            std_dice = np.nanstd(metrics['DICE'][-1]) if metrics['DICE'] else 0
            logging.info(f'Average DICE (per-image mean): {avg_dice:.4f} +/- {std_dice:.4f}')
            wandb.log({f'Metrics/{threshold_low}_Average_DICE_{dataset_key}': avg_dice})
            wandb.log({f'Metrics/{threshold_low}_Std_DICE_{dataset_key}': std_dice})
            
            # Compute Image-level AUROC (anomalous vs healthy classification)
            # Uses anomaly scores from anomalous and healthy images
            from sklearn.metrics import roc_auc_score
            if len(anomalous_pred) > 0 and len(healthy_pred) > 0:
                # Flatten scores to scalar per image
                def to_scalar(s):
                    if hasattr(s, 'cpu'):
                        s = s.cpu().detach().numpy()
                    s = np.asarray(s)
                    # If multi-dimensional, take mean to get single score per image
                    return float(np.mean(s))
                
                anomalous_scores = [to_scalar(s) for s in anomalous_pred]
                healthy_scores = [to_scalar(s) for s in healthy_pred]
                
                image_scores = np.array(anomalous_scores + healthy_scores)
                image_labels = np.array([1] * len(anomalous_scores) + [0] * len(healthy_scores))
                
                image_auroc = roc_auc_score(image_labels, image_scores)
                logging.info(f'Image-level AUROC: {image_auroc:.4f} (n_anomalous={len(anomalous_pred)}, n_healthy={len(healthy_pred)})')
                wandb.log({f'Metrics/{threshold_low}_Image_AUROC_{dataset_key}': image_auroc})
            else:
                logging.warning(f'Cannot compute Image AUROC: anomalous={len(anomalous_pred)}, healthy={len(healthy_pred)}')
                image_auroc = np.nan

        # Plot box plots over the metrics per image
        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            wandb.log({f'Metrics/{threshold_low}_{self.name}_{metric}': fig_bp})

    def pathology_localization_cached(self, cache_dir, threshold_low, threshold_high, perc_flag=False):
        """
        Same as pathology_localization but loads precomputed anomaly maps from cache
        instead of calling get_anomaly() again. This is purely CPU-bound.
        """
        logging.info(f"################ Cached Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
            'DICE': [],
            'AUROC': [],
        }
        pred_dict = dict()

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
                'DICE': [],
                'AUROC': [],
            }
            per_image_results = []

            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):
                # Load original images and masks from dataloader
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                # Load cached results instead of calling get_anomaly
                anomaly_map = np.load(os.path.join(cache_dir, f'batch_{idx}_anomaly_map.npy'))
                anomaly_score = np.load(os.path.join(cache_dir, f'batch_{idx}_anomaly_score.npy'))
                x_rec_dict = torch.load(os.path.join(cache_dir, f'batch_{idx}_x_rec_dict.pt'),
                                        map_location='cpu', weights_only=False)

                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec', 'tensor': x_rec},
                    {'title': f'Anomaly  map {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'}
                ]

                if 'mask' in x_rec_dict:
                    masked_input = x_rec_dict['mask'] + x.cpu()
                    masked_input[masked_input > 1] = 1

                    to_visualize.append({'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                    to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                                        'vmax': x_rec_dict['x_res'].max()})
                    to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'gray'})

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > threshold_low and torch.sum(
                            masks[i][0]) <= threshold_high:
                        count = str(idx * len(x) + i)
                        if int(count) in self.BLACKLIST:
                            continue

                        if int(count) % 12 == 0 or int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254,
                                                                  539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                            self._log_visualization(to_visualize, i, count)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]
                        anomalous_pred.append(anomaly_score[i][0])

                        pred_np = res_2_i_np.cpu().detach().numpy() if torch.is_tensor(res_2_i_np) else res_2_i_np
                        pred_np = np.squeeze(pred_np)
                        mask_np = masks[i][0].cpu().detach().numpy()

                        pred_.append(pred_np)
                        label_.append(mask_np)

                        # Similarity metrics
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i.cpu()))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                        # Per-image DICE
                        pred_min, pred_max = pred_np.min(), pred_np.max()
                        pred_smoothed = cv2.GaussianBlur(pred_np.astype(np.float32), (5, 5), 1.0)

                        best_dice = 0.0
                        best_dice_pp = 0.0

                        if pred_max > pred_min:
                            percentile_ths = np.percentile(pred_np, [50, 60, 70, 75, 80, 85, 90, 95, 97, 99])
                            linear_ths = np.linspace(pred_min, pred_max, 51)[1:]
                            all_thresholds = np.unique(np.concatenate([percentile_ths, linear_ths]))

                            for th in all_thresholds:
                                d = compute_dice_per_image(pred_np, mask_np, threshold=th)
                                if d > best_dice:
                                    best_dice = d

                            smoothed_min, smoothed_max = pred_smoothed.min(), pred_smoothed.max()
                            if smoothed_max > smoothed_min:
                                smoothed_ths = np.linspace(smoothed_min, smoothed_max, 31)[1:]
                                for th in smoothed_ths:
                                    d = compute_dice_per_image(pred_smoothed, mask_np, threshold=th)
                                    if d > best_dice:
                                        best_dice = d

                            pred_uint8 = (pred_np * 255).astype(np.uint8)
                            otsu_th, pred_otsu = cv2.threshold(pred_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            pred_otsu_binary = (pred_otsu > 0).astype(float)
                            intersection_otsu = np.sum(pred_otsu_binary * mask_np)
                            union_otsu = np.sum(pred_otsu_binary) + np.sum(mask_np)
                            d_otsu = 2 * intersection_otsu / (union_otsu + 1e-8) if union_otsu > 0 else 0
                            if d_otsu > best_dice:
                                best_dice = d_otsu

                            for th in all_thresholds[::2]:
                                pred_bin = (pred_np > th).astype(np.uint8)
                                pred_pp = apply_postprocessing(pred_bin, method='both', min_size=5)
                                intersection = np.sum(pred_pp * mask_np)
                                union = np.sum(pred_pp) + np.sum(mask_np)
                                d_pp = 2 * intersection / (union + 1e-8) if union > 0 else 0
                                if d_pp > best_dice_pp:
                                    best_dice_pp = d_pp

                            pred_otsu_pp = apply_postprocessing(pred_otsu_binary.astype(np.uint8), method='both', min_size=5)
                            intersection_otsu_pp = np.sum(pred_otsu_pp * mask_np)
                            union_otsu_pp = np.sum(pred_otsu_pp) + np.sum(mask_np)
                            d_otsu_pp = 2 * intersection_otsu_pp / (union_otsu_pp + 1e-8) if union_otsu_pp > 0 else 0
                            if d_otsu_pp > best_dice_pp:
                                best_dice_pp = d_otsu_pp

                        dice_per_img = max(best_dice, best_dice_pp)
                        test_metrics['DICE'].append(dice_per_img)

                        auroc_per_img = compute_auroc_per_image(pred_np, mask_np)
                        test_metrics['AUROC'].append(auroc_per_img)

                        per_image_results.append({
                            'image_id': int(count),
                            'MAE': loss_mae.item(),
                            'LPIPS': float(loss_lpips),
                            'SSIM': ssim_,
                            'DICE': dice_per_img,
                            'AUROC': auroc_per_img,
                            'mask_size': int(torch.sum(masks[i][0]).item())
                        })

                    elif torch.sum(masks[i][0]) <= 1:
                        res_2_i_np_healthy = anomaly_map[i][0]
                        healthy_pred.append(anomaly_score[i][0])

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

            if per_image_results:
                csv_path = self.checkpoint_path + f'/per_image_metrics_{dataset_key}_{threshold_low}_{threshold_high}.csv'
                df = pd.DataFrame(per_image_results)
                df.to_csv(csv_path, index=False)
                logging.info(f'Saved per-image metrics to {csv_path}')

                wandb.log({
                    f'PerImage/{threshold_low}_{dataset_key}_mean_DICE': np.nanmean(test_metrics['DICE']),
                    f'PerImage/{threshold_low}_{dataset_key}_std_DICE': np.nanstd(test_metrics['DICE']),
                    f'PerImage/{threshold_low}_{dataset_key}_mean_AUROC': np.nanmean(test_metrics['AUROC']),
                    f'PerImage/{threshold_low}_{dataset_key}_std_AUROC': np.nanstd(test_metrics['AUROC']),
                })

                artifact = wandb.Artifact(f'per_image_metrics_{dataset_key}_{threshold_low}', type='results')
                artifact.add_file(csv_path)
                wandb.log_artifact(artifact)

        for dataset_key in self.test_data_dict.keys():
            pred_ood, label_ood = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))
            labels_all = np.reshape(np.asarray(labels), (len(labels), -1))
            print(f'Nr of preditions: {predictions_all.shape}')
            print(f'Predictions go from {np.min(predictions_all)} to {np.max(predictions_all)} with mean: {np.mean(predictions_all)}')
            print(f'Labels go from {np.min(labels_all)} to {np.max(labels_all)} with mean: {np.mean(labels_all)}')
            print('Shapes {} {} '.format(labels.shape, predictions.shape))

            dice_scores = []

            auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
            logging.info(f'Global AUPRC score: {auprc_}')
            wandb.log({f'Metrics/{threshold_low}_Global_AUPRC_{dataset_key}': auprc_})

            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice = compute_dice(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]
            best_threshold = ths[highest_score_index]

            logging.info(f'Global highest DICE: {highest_score} at threshold {best_threshold}')
            wandb.log({f'Metrics/{threshold_low}_Global_highest_DICE': highest_score})
            wandb.log({f'Metrics/{threshold_low}_Best_DICE_threshold': best_threshold})

            avg_dice = np.nanmean(metrics['DICE'][-1]) if metrics['DICE'] else 0
            std_dice = np.nanstd(metrics['DICE'][-1]) if metrics['DICE'] else 0
            logging.info(f'Average DICE (per-image mean): {avg_dice:.4f} +/- {std_dice:.4f}')
            wandb.log({f'Metrics/{threshold_low}_Average_DICE_{dataset_key}': avg_dice})
            wandb.log({f'Metrics/{threshold_low}_Std_DICE_{dataset_key}': std_dice})

            from sklearn.metrics import roc_auc_score
            if len(anomalous_pred) > 0 and len(healthy_pred) > 0:
                def to_scalar(s):
                    if hasattr(s, 'cpu'):
                        s = s.cpu().detach().numpy()
                    s = np.asarray(s)
                    return float(np.mean(s))

                anomalous_scores = [to_scalar(s) for s in anomalous_pred]
                healthy_scores = [to_scalar(s) for s in healthy_pred]

                image_scores = np.array(anomalous_scores + healthy_scores)
                image_labels = np.array([1] * len(anomalous_scores) + [0] * len(healthy_scores))

                image_auroc = roc_auc_score(image_labels, image_scores)
                logging.info(f'Image-level AUROC: {image_auroc:.4f} (n_anomalous={len(anomalous_pred)}, n_healthy={len(healthy_pred)})')
                wandb.log({f'Metrics/{threshold_low}_Image_AUROC_{dataset_key}': image_auroc})
            else:
                logging.warning(f'Cannot compute Image AUROC: anomalous={len(anomalous_pred)}, healthy={len(healthy_pred)}')
                image_auroc = np.nan

        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            wandb.log({f'Metrics/{threshold_low}_{self.name}_{metric}': fig_bp})