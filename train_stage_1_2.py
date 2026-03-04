#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from boxmot import BotSort


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image-pair segmentation tracking")
    parser.add_argument('--source', type=str, required=True, help='Image folder path')
    parser.add_argument('--all_seq_mask_path', type=str, required=True, help='GT mask folder path')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--cam', type=int, default=0, help='Camera index')
    parser.add_argument('--output', type=str, default='output_pairs', help='Output folder path')
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt', help='ReID weights path')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--min-bbox-area', type=int, default=500, help='Minimum bbox area (pixels)')
    parser.add_argument('--score-threshold', type=float, default=0.001, help='Fixed threshold for cubic scores')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--save-npy', action='store_true', help='Save tracking results to npy')
    return parser.parse_args()


def get_color(track_id):
    """Generate a unique color for each track ID."""
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def find_image_pairs(source_path, cam_id):
    """Find all image pairs.

    Args:
        source_path: Image folder path

    Returns:
        pairs: [(gt_path, warp_path, frame_id, cam_id), ...]
    """
    source_path = Path(source_path)
    pairs = []
    
    # Find all gt images
    print(f"Searching cam{cam_id} images in {source_path}...")
    gt_pattern = re.compile(r'gt_(\d+)_cam(\d+)\.png')
    gt_files = list(source_path.glob('gt_*_cam*.png'))
    
    for gt_file in gt_files:
        match = gt_pattern.match(gt_file.name)
        if match:
            frame_id, cam = match.groups()
            if int(cam) == int(cam_id):
                # Find matching warp image
                warp_file = source_path / f'warp_{frame_id}_cam{cam_id}.png'
                if warp_file.exists():
                    pairs.append((gt_file, warp_file, frame_id, cam_id))
        
    print(f"Found {len(pairs)} image pairs")
    
    return sorted(pairs, key=lambda x: (int(x[2]), int(x[3])))  # sort by frame_id and cam_id


def process_image(image_path, segmentation_model, device, conf_thres):
    """Process a single image for detection and segmentation.

    Returns:
        dets: detection array
        masks: list of masks
        im: original image
    """
    # Read image
    im = cv2.imread(str(image_path))
    if im is None:
        print(f"Failed to read image {image_path}")
        return None, None, None
    
    # Convert image to tensor and move to device
    frame_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    
    # Run Mask R-CNN for boxes and masks
    with torch.no_grad():
        results = segmentation_model(frame_tensor)[0]
    
    # Extract results
    dets = []
    masks = [] 
    track_class = [1, 2, 3, 4, 6, 7, 8]
    
    for i, score in enumerate(results['scores']):
        if score >= conf_thres:
            # Extract box and score
            x1, y1, x2, y2 = results['boxes'][i].cpu().numpy()
            conf = score.item()
            cls = results['labels'][i].item()  
            if cls not in track_class:
                continue
            dets.append([x1, y1, x2, y2, conf, cls])
            
            # Extract mask (first channel)
            mask = results['masks'][i, 0].cpu().numpy()
            masks.append(mask)
    
    # Convert detections to numpy array
    if dets:
        dets = np.array(dets)
    else:
        dets = np.empty((0, 6))
    
    return dets, masks, im


def create_tracking_mask(tracks, masks, image_shape):
    """Create a tracking mask.

    Returns:
        track_mask: tracking mask array
        track_info: {track_id: area} dict
    """
    track_mask = np.ones(image_shape[:2], dtype=np.int32) * -1
    track_info = {}
    
    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')  # tracking indices
        
        # Ensure indices are in range
        valid_masks = []
        for i in inds:
            if i < len(masks):
                valid_masks.append(masks[i])
            else:
                valid_masks.append(None)
        
        # Traverse tracks and corresponding masks
        for track, mask in zip(tracks, valid_masks):
            track_id = int(track[4])
            
            if mask is not None:
                # Binarize mask
                binary_mask = (mask > 0.5).astype(np.uint8)
                active_pixels = np.sum(binary_mask)
                
                if active_pixels > 0:
                    # Fill mask region with track ID
                    track_mask[binary_mask == 1] = track_id
                    track_info[track_id] = active_pixels
    
    return track_mask, track_info


def calculate_motion_metrics(bbox1, bbox2):
    """Compute a more sensitive motion metric."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Skip sudden bbox changes
    horizon_change_ratio = abs((x2_2 - x1_2) - (x2_1 - x1_1))/abs(x2_1 - x1_1)
    vertical_change_ratio = abs((y2_2 - y1_2) - (y2_1 - y1_1))/abs(y2_1 - y1_1)
    if horizon_change_ratio > 0.3 or vertical_change_ratio > 0.3:
        return None
    

    
    # 1. Center displacement
    center1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
    center2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])
    center_displacement = np.linalg.norm(center2 - center1)

    
    # 2. Area change ratio
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_change_ratio = abs(area2 - area1) / area1 if area1 > 0 else 0
    
    # 3. Aspect ratio change
    aspect_ratio1 = (x2_1 - x1_1) / (y2_1 - y1_1) if (y2_1 - y1_1) > 0 else 0
    aspect_ratio2 = (x2_2 - x1_2) / (y2_2 - y1_2) if (y2_2 - y1_2) > 0 else 0
    aspect_ratio_change = abs(aspect_ratio2 - aspect_ratio1)
    if aspect_ratio_change > 0.15:
        return None
    
    
    # 4. Edge displacement std (shape change)
    edges1 = np.array([x1_1, y1_1, x2_1, y2_1])
    edges2 = np.array([x1_2, y1_2, x2_2, y2_2])
    edge_displacements = np.abs(edges2 - edges1)
    shape_variance = np.std(edge_displacements)
    
    # 5. Motion intensity (combined)
    bbox_size = np.sqrt(area1)  # normalize displacement
    normalized_displacement = center_displacement / bbox_size if bbox_size > 0 else 0

    
    # Combine center displacement, area change, and shape change
    motion_intensity = (
        0.8 * normalized_displacement +
        0.1 * area_change_ratio +
        0.1 * shape_variance / bbox_size if bbox_size > 0 else 0
    )
    
    return motion_intensity


def filter_small_bboxes(tracks, min_area):
    """Filter out bboxes with area below a threshold."""
    if len(tracks) == 0:
        return tracks
    
    filtered_tracks = []
    for track in tracks:
        x1, y1, x2, y2 = track[:4]
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area >= min_area:
            filtered_tracks.append(track)
    
    return np.array(filtered_tracks) if filtered_tracks else np.empty((0, tracks.shape[1]))


def get_mask_by_track_id(tracks, masks, track_id):
    """Get the mask for a given track ID."""
    for i, track in enumerate(tracks):
        if int(track[4]) == track_id:
            # tracking index
            mask_idx = int(track[7])
            if mask_idx < len(masks):
                return masks[mask_idx]
    return None


def get_bbox_by_track_id(tracks, track_id):
    """Get the bbox for a given track ID."""
    for track in tracks:
        if int(track[4]) == track_id:
            return track[:4].astype('float')
    return None


def visualize_tracking_result(image, tracks, masks, id_mapping, rgb_diff_vis):
    """Visualize tracking results."""
    vis_image = image.copy()
    rgb_diff_vis = rgb_diff_vis.copy()
    
    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')
        
        # Ensure indices are in range
        valid_masks = []
        for i in inds:
            if i < len(masks):
                valid_masks.append(masks[i])
            else:
                valid_masks.append(None)
        
        # Draw masks and boxes
        for track, mask in zip(tracks, valid_masks):
            track_id = int(track[4])
            if track_id not in id_mapping.keys():
                continue
            else:
                color = get_color(id_mapping[track_id]) 
            
            # Draw segmentation mask
            if mask is not None:
                binary_mask = (mask > 0.5).astype(np.uint8)
                if np.sum(binary_mask) > 0:
                    # Blend mask color into image
                    vis_image[binary_mask == 1] = vis_image[binary_mask == 1] * 0.5 + np.array(color) * 0.5
                    rgb_diff_vis[binary_mask == 1] = rgb_diff_vis[binary_mask == 1] * 0.5 + np.array(color) * 0.5
            
            # Draw bbox
            x1, y1, x2, y2 = track[:4].astype('int')
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(rgb_diff_vis, (x1, y1), (x2, y2), color, 2)
            
            # Add ID label
            cv2.putText(vis_image, f'ID: {id_mapping[track_id]}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(rgb_diff_vis, f'ID: {id_mapping[track_id]}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image, rgb_diff_vis


def visualize_pair_results(
    output_path,
    warp_dir,
    frame_id,
    cam_id,
    gt_image,
    warp_image,
    gt_tracks,
    warp_tracks,
    gt_masks,
    warp_masks,
    id_mapping,
    rgb_diff,
):
    """Render and save a 2x2 visualization for a GT/warp pair."""
    rgb_diff_vis = (rgb_diff * 255 / np.max(rgb_diff)).astype(np.uint8)
    rgb_diff_vis = cv2.applyColorMap(rgb_diff_vis, cv2.COLORMAP_JET)
    rgb_diff_vis = cv2.resize(rgb_diff_vis, (gt_image.shape[1], gt_image.shape[0]))

    gt_vis, gt_diff_vis = visualize_tracking_result(
        gt_image,
        gt_tracks,
        gt_masks,
        id_mapping,
        rgb_diff_vis,
    )
    warp_vis, warp_diff_vis = visualize_tracking_result(
        warp_image,
        warp_tracks,
        warp_masks,
        id_mapping,
        rgb_diff_vis,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    gt_vis_rgb = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)
    warp_vis_rgb = cv2.cvtColor(warp_vis, cv2.COLOR_BGR2RGB)
    gt_diff_vis_rgb = cv2.cvtColor(gt_diff_vis, cv2.COLOR_BGR2RGB)
    warp_diff_vis_rgb = cv2.cvtColor(warp_diff_vis, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(gt_vis_rgb)
    axes[0, 0].set_title(f"GT Frame {frame_id}", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(warp_vis_rgb)
    axes[0, 1].set_title(f"Warp Frame {frame_id}", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gt_diff_vis_rgb)
    axes[1, 0].set_title(f"GT Diff Frame {frame_id}", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(warp_diff_vis_rgb)
    axes[1, 1].set_title(f"Warp Diff Frame {frame_id}", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()

    os.makedirs(output_path / warp_dir, exist_ok=True)
    combined_vis_path = output_path / warp_dir / f"warp_{frame_id}_cam{cam_id}_diff.png"
    plt.savefig(str(combined_vis_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_id_mapping(mask_A, mask_B, min_iou=0.5):
    """Create an ID mapping from mask_B to mask_A using IoU."""
    # Unique non-background IDs
    ids_A = np.unique(mask_A[mask_A >= 0])
    ids_B = np.unique(mask_B[mask_B >= 0])
    
    mapping_dict = {}
    used_A_ids = set()
    
    for id_B in ids_B:
        mask_B_region = (mask_B == id_B)
        best_match_id = -1
        best_iou = 0
        
        for id_A in ids_A:
            if id_A in used_A_ids:
                continue
                
            mask_A_region = (mask_A == id_A)
            
            # Compute IoU
            intersection = np.sum(mask_A_region & mask_B_region)
            union = np.sum(mask_A_region | mask_B_region)
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou and iou >= min_iou:
                best_iou = iou
                best_match_id = id_A
        
        if best_match_id != -1:
            mapping_dict[id_B] = best_match_id
            used_A_ids.add(best_match_id)
            # print(f"Map: mask_B ID {id_B} -> mask_A ID {best_match_id} (IoU: {best_iou:.3f})")
    
    return mapping_dict


def visualize_dynamic_scores(score_dict, threshold=None, output_path=None):
    """Plot cubic scores as a line chart and optionally save it."""
    if not score_dict:
        print("No scores to visualize")
        return

    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    ids = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    plt.figure(figsize=(14, 8))
    plt.plot(
        range(len(ids)),
        scores,
        marker='o',
        linewidth=2,
        markersize=6,
        color='#1f77b4',
        markerfacecolor='#ff7f0e',
        markeredgecolor='#1f77b4',
    )

    if threshold is not None:
        plt.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            alpha=0.7,
            linewidth=2,
            label=f'Threshold: {threshold:.4f}',
        )
        plt.annotate(
            f'Threshold\n{threshold:.4f}',
            xy=(len(ids) * 0.8, threshold),
            xytext=(len(ids) * 0.8, threshold + max(scores) * 0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
        )
        plt.legend(loc='upper right')

    plt.title('Cubic Motion Scores by Track ID', fontsize=16, fontweight='bold')
    plt.xlabel('Track ID Rank (Sorted by Score)', fontsize=12)
    plt.ylabel('Cubic Motion Score', fontsize=12)
    plt.xticks(range(len(ids)), [str(track_id) for track_id in ids], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, linestyle='--')

    for i in range(min(10, len(ids))):
        plt.annotate(
            f'{scores[i]:.2f}',
            (i, scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dynamic scores plot saved to: {output_path}")

    print("\nStatistics:")
    print(f"Total number of tracked objects: {len(ids)}")
    print(f"Highest cubic score: {max(scores):.4f} (ID: {ids[0]})")
    print(f"Lowest cubic score: {min(scores):.4f} (ID: {ids[-1]})")
    print(f"Average cubic score: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")

    if threshold is not None:
        above_threshold = [score for score in scores if score >= threshold]
        print(f"Threshold: {threshold:.4f}")
        print(f"Objects above threshold: {len(above_threshold)}/{len(scores)}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Set device (with CUDA availability check)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Load Mask R-CNN
    segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')    
    segmentation_model.eval().to(device)
    print("Loaded Mask R-CNN model")
    
    # Find all image pairs
    seq_ID_score = {}  # dynamic scores after ID mapping
    seq_ID_score_count = {}  # observation counts after ID mapping
    mapping_list = os.listdir(args.source)
    mapping_list = [p for p in mapping_list if p.startswith('warp_') and not (p.endswith('.tar'))] 
    mapping_list.sort(key=lambda x: int(x.split('_')[1]))
    mapping_list = mapping_list[args.start_frame:]
    # mapping_list = [warp_0, warp_1, ...]
    for warp_dir in mapping_list:
        image_pairs = find_image_pairs(os.path.join(args.source, warp_dir), args.cam)
        # N * (gt_path, warp_path, frame_id, cam_id)
        
        if not image_pairs:
            print(f"No matched image pairs found in {warp_dir}")
            return
        
        
        # Process each image pair
        for gt_path, warp_path, frame_id, cam_id in image_pairs:
            print(f"\nProcessing Mapping {warp_dir.split('_')[1]} - warp {frame_id}- Cam {cam_id}")

            # Initialize tracker (new tracker per image pair)
            tracker = BotSort(
                reid_weights=Path(args.reid_weights),
                device=device,
                half=False,
            )
            
            # Process GT image (first frame)
            gt_dets, gt_masks, gt_image = process_image(gt_path, segmentation_model, device, args.conf_thres)
            if gt_image is None:
                continue
            
            # Update tracker - GT frame
            gt_tracks = tracker.update(gt_dets, gt_image)
            
            # Filter small bboxes
            gt_tracks_filtered = filter_small_bboxes(gt_tracks, args.min_bbox_area)
            
            gt_track_mask, gt_track_info = create_tracking_mask(gt_tracks_filtered, gt_masks, gt_image.shape)
            
            # Process warp image (second frame)
            warp_dets, warp_masks, warp_image = process_image(warp_path, segmentation_model, device, args.conf_thres)
            if warp_image is None:
                continue
            
            # Update tracker - warp frame
            warp_tracks = tracker.update(warp_dets, warp_image)
            
            # Filter small bboxes
            warp_tracks_filtered = filter_small_bboxes(warp_tracks, args.min_bbox_area)
            
            
            warp_track_mask, warp_track_info = create_tracking_mask(warp_tracks_filtered, warp_masks, warp_image.shape)
            
            # Compute motion intensity
            gt_ids = set(gt_track_info.keys())
            warp_ids = set(warp_track_info.keys())
            common_ids = gt_ids & warp_ids
            rgb_diff = np.abs(gt_image.astype(np.float32) - warp_image.astype(np.float32)).mean(axis=2)  # (H,W,3)=> (H,W)
            thre  = np.percentile(rgb_diff, 98)  # 98th percentile threshold
            
            # Create binary mask and erode to reduce noise
            binary_mask = (rgb_diff >= thre).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3x3 ellipse kernel
            binary_mask_eroded = cv2.erode(binary_mask, kernel, iterations=1)
            
            # Apply eroded mask to rgb_diff
            # rgb_diff = rgb_diff * binary_mask_eroded
            
            if common_ids:
                # Compute motion metrics for each common ID
                pair_metrics = {}
                pair_metrics_count = {}
                for track_id in common_ids:
                    # Get masks for this ID in both frames
                    gt_mask = get_mask_by_track_id(gt_tracks_filtered, gt_masks, track_id)
                    warp_mask = get_mask_by_track_id(warp_tracks_filtered, warp_masks, track_id)

                    gt_bbox = get_bbox_by_track_id(gt_tracks_filtered, track_id)
                    warp_bbox = get_bbox_by_track_id(warp_tracks_filtered, track_id)

                    x1_1, y1_1, x2_1, y2_1 = gt_bbox
                    x1_2, y1_2, x2_2, y2_2 = warp_bbox

                    # Skip sudden bbox changes
                    horizon_change_ratio = abs((x2_2 - x1_2) - (x2_1 - x1_1))/abs(x2_1 - x1_1)
                    vertical_change_ratio = abs((y2_2 - y1_2) - (y2_1 - y1_1))/abs(y2_1 - y1_1)
                    if horizon_change_ratio > 0.3 or vertical_change_ratio > 0.3:
                        continue

                    motion_intensity_1 = calculate_motion_metrics(gt_bbox, warp_bbox)
                   
                    if gt_mask is not None and warp_mask is not None:
                        # Binarize masks
                        gt_binary_mask = (gt_mask > 0.5).astype(np.uint8)
                        warp_binary_mask = (warp_mask > 0.5).astype(np.uint8)
                        
                        # Compute mask areas
                        gt_mask_area = np.sum(gt_binary_mask)
                        warp_mask_area = np.sum(warp_binary_mask)
                        
                        if gt_mask_area > 0 and warp_mask_area > 0:
                            # Compute RGB diff within masks
                            gt_mask_diff = binary_mask_eroded[gt_binary_mask == 1].sum()/ gt_mask_area
                            warp_mask_diff = binary_mask_eroded[warp_binary_mask == 1].sum()/ warp_mask_area
                            motion_intensity = gt_mask_diff + warp_mask_diff
                          
                            
                            if motion_intensity is not None:
                                if motion_intensity_1 is not None:
                                    motion_intensity += motion_intensity_1
                                pair_metrics[track_id] = motion_intensity
                                pair_metrics_count[track_id] = pair_metrics_count.get(track_id, 0) + 1
                        else:
                            print(f"  ID {track_id}: mask area is 0")
                    else:
                        print(f"  ID {track_id}: failed to get mask")
                

            # ID matching
            seq_mask_paths = os.listdir(args.all_seq_mask_path)
            seq_mask_paths = sorted([p for p in seq_mask_paths if p.endswith('.npy')])
            seq_mask_paths = seq_mask_paths[args.start_frame:]
            seq_mask_path = seq_mask_paths[int(frame_id)-args.start_frame]
            # print(f"Matching warp_{frame_id} back to seq_mask: {seq_mask_path}")
            # Create ID mapping
            all_seq_mask = np.load(os.path.join(args.all_seq_mask_path, seq_mask_path))
            all_seq_mask = cv2.resize(all_seq_mask, (gt_track_mask.shape[1], gt_track_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            id_mapping = create_id_mapping(all_seq_mask, gt_track_mask)
            # Update per-warp dynamic scores
            pair_metrics_updated = {}
            pair_metrics_count_updated = {}
            for track_id, motion_intensity in pair_metrics.items():
                if track_id in id_mapping:
                    mapped_id = id_mapping[track_id]
                    pair_metrics_updated[mapped_id] = motion_intensity
                    pair_metrics_count_updated[mapped_id] = pair_metrics_count.get(track_id, 0)
                    # print(f"ID {track_id} mapped to {mapped_id}, motion: {motion_intensity:.4f}")
                else:
                    print(f"ID {track_id} did not map to any ID")
            # Update seq_ID_score
            for track_id, motion_intensity in pair_metrics_updated.items():
                if track_id not in seq_ID_score:
                    seq_ID_score[track_id] = motion_intensity
                    seq_ID_score_count[track_id] = pair_metrics_count_updated.get(track_id, 0)
                else:
                    seq_ID_score[track_id] += motion_intensity
                    seq_ID_score_count[track_id] += pair_metrics_count_updated.get(track_id, 0)

            visualize_pair_results(
                output_path,
                warp_dir,
                frame_id,
                cam_id,
                gt_image,
                warp_image,
                gt_tracks_filtered,
                warp_tracks_filtered,
                gt_masks,
                warp_masks,
                id_mapping,
                rgb_diff,
            )
            


       

    print(f"\n================= Final ID scores =================")
    final_scores = []  # final score list
    for track_id, score in sorted(seq_ID_score.items()):
        # Observed at least 6 times
        if seq_ID_score_count.get(track_id, 1) > 6:
            final_scores.append((track_id, score / seq_ID_score_count.get(track_id, 1)))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    # for track_id, score in final_scores:
    #     print(f"ID {track_id}: dynamic score {score:.4f} (count: {seq_ID_score_count.get(track_id, 1)})")
    
    # Save scores to file
    score_file = output_path.parent / "dynamic_scores.txt"
    with open(score_file, 'w') as f:
        f.write("ID\tdynamic_score\n")
        for track_id, score in final_scores:
            f.write(f"{track_id}\t{score:.4f}\n")
    print(f"Dynamic scores saved to {score_file}")

    if final_scores:
        final_score_dict = {track_id: score for track_id, score in final_scores}
        cubic_score_dict = {track_id: score ** 3 for track_id, score in final_scores}
        plot_path = output_path.parent / "cubic_scores_line_plot.png"
        visualize_dynamic_scores(cubic_score_dict, args.score_threshold, str(plot_path))

        above_threshold_ids = [
            int(track_id) for track_id, score in cubic_score_dict.items()
            if score >= args.score_threshold
        ]
        above_threshold_ids.sort()
        json_path = output_path.parent / "dynamic_ids.json"
        json_payload = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    json_payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                json_payload = {}

        json_payload[f"cam_{args.cam}"] = above_threshold_ids
        with open(json_path, 'w') as f:
            json.dump(json_payload, f, indent=4)
        print(f"Cubic-score ID list saved to {json_path}")
    else:
        print("No final scores available; skipping plot and JSON generation")



if __name__ == "__main__":
    main()
