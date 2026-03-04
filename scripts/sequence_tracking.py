#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image sequence segmentation and tracking example.

Processes image sequences in a folder for instance segmentation and object tracking.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from boxmot import BotSort


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Image sequence segmentation and tracking")
    parser.add_argument('--source', type=str, required=True, help='Image folder path')
    parser.add_argument('--output', type=str, default='output', help='Output folder path')
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt', help='ReID weights path')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device, cpu or cuda')
    parser.add_argument('--save-vid', action='store_true', help='Save video output')
    parser.add_argument('--vid-fps', type=int, default=15, help='Video FPS')
    parser.add_argument('--save-npy', action='store_true', help='Save tracking results as npy')
    parser.add_argument('--npy-path', type=str, default='npy_masks', help='NPY output path')
    return parser.parse_args()


def get_color(track_id):
    """Generate a unique color for each track ID."""
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def visualize_mask(mask_array):
    """Visualize an ID mask as a color image.

    Args:
        mask_array: 2D array with values -1 (background) or track IDs

    Returns:
        colored_mask: Color visualization mask in RGB format
    """
    # Create a black RGB image.
    h, w = mask_array.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get all unique IDs (excluding background -1).
    unique_ids = np.unique(mask_array)
    unique_ids = unique_ids[unique_ids >= 0]
    
    # Color each ID.
    for track_id in unique_ids:
        # Get the color for this ID.
        color = get_color(int(track_id))
        # Paint the region for this ID.
        colored_mask[mask_array == track_id] = color
    
    return colored_mask


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory.
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create npy output directory.
    if args.save_npy:
        npy_path = Path(args.npy_path)
        npy_path.mkdir(exist_ok=True, parents=True)
        print(f"Tracking masks will be saved as npy files to {npy_path}")
    
    # Set device.
    device = torch.device(args.device)
    
    # Load segmentation model.
    segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')    
    segmentation_model.eval().to(device)
    
    # Initialize tracker.
    print(f"Initializing BotSort with ReID weights: {args.reid_weights}")
    tracker = BotSort(
        reid_weights=Path(args.reid_weights),
        device=device,
        half=False,
    )
    
    # Collect all image files.
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(args.source).glob(f'*{ext}')))
        image_files.extend(list(Path(args.source).glob(f'*{ext.upper()}')))
    
    # Sort by filename.
    image_files = sorted(image_files)
    base_names = [f.name.split('.')[0] for f in image_files]
    
    if not image_files:
        print(f"No images found in {args.source}")
        return
    
    # Read the first image to get dimensions.
    first_img = cv2.imread(str(image_files[0]))
    img_height, img_width = first_img.shape[:2]
    
    # Video writer.
    video_writer = None
    if args.save_vid:
        video_path = output_path / 'tracking_result.mp4'
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            args.vid_fps,
            (img_width, img_height)
        )
    
    track_class = [1,2,3,4,6,7,8]
    # Process each image.
    for idx, img_path in enumerate(tqdm(image_files, desc="处理图片")):
        # Read image.
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"Failed to read image {img_path}")
            continue
        
        # Convert image to tensor and move to device.
        frame_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0).to(device)
        
        # Run Mask R-CNN to get boxes and masks.
        with torch.no_grad():
            results = segmentation_model(frame_tensor)[0]

            # 'box': (N,4) bounding boxes
            # 'labels': (N,) class labels
            # 'scores': (N,) confidence scores
            # 'masks': (N,1,H,W) segmentation masks
        
        # Extract segmentation results.
        dets = []
        masks = [] 
        confidence_threshold = args.conf_thres
        
        for i, score in enumerate(results['scores']):
            if score >= confidence_threshold:
                # Extract bounding box and score.
                x1, y1, x2, y2 = results['boxes'][i].cpu().numpy()
                conf = score.item()
                cls = results['labels'][i].item()  
                if cls not in track_class:
                    continue
                dets.append([x1, y1, x2, y2, conf, cls])
                
                # Extract mask and add to list.
                mask = results['masks'][i, 0].cpu().numpy()  # Use first channel (binary mask).
                masks.append(mask)
        
        # Convert detections to numpy array (N x (x, y, x, y, conf, cls)).
        if dets:
            dets = np.array(dets)
        else:
            dets = np.empty((0, 6))
        
        # Update tracker.
        tracks = tracker.update(dets, im) 
        # tracks format: M x (x1, y1, x2, y2, id, conf, cls, ind)
        

        # id is the unique identifier assigned by the tracker
        # ind is the index in dets (and masks) corresponding to each track
        
        if args.save_npy:
            # Create an array matching image size, initialized to -1.
            track_mask = np.ones((im.shape[0], im.shape[1]), dtype=np.int32) * -1
       
        # Draw masks and boxes in a single pass.
        if len(tracks) > 0:
            inds = tracks[:, 7].astype('int')  # Track indices as integers.
            
            
            # Match tracks to masks by index.
            if len(masks) > 0:
                # Ensure indices are valid.
                valid_masks = []
                for i in inds:
                    if i < len(masks):
                        # This object has a valid mask for the frame.
                        valid_masks.append(masks[i])
                    else:
                        # This object has no valid mask for the frame.
                        # Add an empty placeholder mask.
                        valid_masks.append(None)
                masks = valid_masks  # Now there are len(tracks) masks.
            
            # Draw each track with its mask.
            for track_idx, (track, mask) in enumerate(zip(tracks, masks)):
                track_id = int(track[4])  # Track ID.
                color = get_color(track_id)  # Unique color per track.
                
                # Draw segmentation mask.
                if mask is not None:

                    # Binarize mask with a lower threshold for more details.
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    # Count active pixels in the binary mask.
                    active_pixels = np.sum(binary_mask)
                    
                    # Update track mask if saving npy.
                    if args.save_npy and active_pixels > 0:
                        # Fill current object ID into the mask.
                        track_mask[binary_mask == 1] = track_id
                        
                    # Blend mask color into the image.
                    im[binary_mask == 1] = im[binary_mask == 1] * 0.5 + np.array(color) * 0.5
                
                # Draw bounding box.
                x1, y1, x2, y2 = track[:4].astype('int')
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                
                # Add ID, confidence, and class text.
                conf = track[5]
                cls = track[6]
                cv2.putText(im, f'ID: {track_id}, Conf: {conf:.2f}, Class: {cls}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save tracking mask as npy.
        if args.save_npy:
            npy_file_path = npy_path / f"{base_names[idx]}.npy"
            np.save(str(npy_file_path), track_mask)
            
            # Visualize the tracking mask and save as image.
            vis_path = npy_path / "visualized"
            vis_path.mkdir(exist_ok=True, parents=True)
            
            # Generate colored visualization.
            colored_mask = visualize_mask(track_mask)
            
            # Save visualization result.
            vis_file_path = vis_path / f"mask_vis_{idx:04d}.jpg"
            cv2.imwrite(str(vis_file_path), colored_mask)
            
            # Count non-background pixels in the mask.
            non_bg_pixels = np.sum(track_mask >= 0)
            print(f"Frame {idx}: mask has {non_bg_pixels} non-background pixels")
            
            # Warn if mask is all background.
            if non_bg_pixels == 0:
                print(f"Warning: frame {idx} mask is all background (-1)")
        
        # Save processed image.
        output_img_path = output_path / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(output_img_path), im)
        
        # Add to video.
        if args.save_vid and video_writer is not None:
            video_writer.write(im)
    
    # Cleanup.
    if video_writer is not None:
        video_writer.release()
    
    cv2.destroyAllWindows()
    print(f"Done. Color results saved to {output_path}, npy masks saved to {npy_path}.")


if __name__ == "__main__":
    main()
