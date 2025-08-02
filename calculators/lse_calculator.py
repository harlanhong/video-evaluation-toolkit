#!/usr/bin/env python3
"""
LSE (Lip-Sync Error) Calculator
SyncNet-based Python API for calculating LSE-D (Distance) and LSE-C (Confidence) metrics

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides Python API for SyncNet-based lip-sync error calculation.

Usage:
    from evalutation.lse_calculator import LSECalculator
    
    calculator = LSECalculator()
    lse_distance, lse_confidence = calculator.calculate_single_video("video.mp4")
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import subprocess
import tempfile
import shutil
import glob
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import python_speech_features
from scipy.io import wavfile
from scipy import signal
from scipy.interpolate import interp1d
import scenedetect
from scenedetect import detect, ContentDetector
from scenedetect.frame_timecode import FrameTimecode

# Import local SyncNet modules
try:
    # Use relative import when imported as package
    from .syncnet_core.model import S
    from .syncnet_core.detectors import S3FD
except ImportError:
    # Use absolute import when run directly
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from syncnet_core.model import S
    from syncnet_core.detectors import S3FD


class LSECalculator:
    """LSE Calculator"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 batch_size: int = 20,
                 vshift: int = 15,
                 facedet_scale: float = 0.25,
                 crop_scale: float = 0.40,
                 min_track: int = 100,
                 frame_rate: int = 25,
                 num_failed_det: int = 25,
                 min_face_size: int = 100):
        """
        Initialize LSE calculator
        
        Args:
            model_path: Path to the SyncNet model. Uses default if None.
            device: Computing device ("cuda" or "cpu").
            batch_size: Batch size for processing.
            vshift: Video shift range.
            facedet_scale: Scale factor for face detection.
            crop_scale: Scale factor for cropping.
            min_track: Minimum duration for tracking.
            frame_rate: Frame rate for video processing.
            num_failed_det: Number of allowed consecutive detection failures.
            min_face_size: Minimum face size in pixels.
        """
        self.device = device
        self.batch_size = batch_size
        self.vshift = vshift
        self.facedet_scale = facedet_scale
        self.crop_scale = crop_scale
        self.min_track = min_track
        self.frame_rate = frame_rate
        self.num_failed_det = num_failed_det
        self.min_face_size = min_face_size
        
        # Determine model path
        if model_path is None:
            # First try main models directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            main_model_path = os.path.join(project_root, "models", "syncnet_v2.model")
            
            if os.path.exists(main_model_path):
                model_path = main_model_path
            else:
                # Fallback to calculators/models (for backward compatibility)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "models", "syncnet_v2.model")
        
        # Initialize SyncNet model
        print(f"🔄 Loading SyncNet model: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SyncNet model file not found: {model_path}")
            
        self.syncnet = S(num_layers_in_fc_layers=1024)
        if self.device == "cuda" and torch.cuda.is_available():
            self.syncnet = self.syncnet.cuda()
        else:
            self.device = "cpu"
            
        # Load model parameters
        loaded_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self_state = self.syncnet.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        self.syncnet.eval()
        
        # Initialize face detector
        print(f"🔄 Initializing face detector")
        try:
            self.face_detector = S3FD(device=self.device)
            print(f"✅ LSE calculator initialized (Device: {self.device})")
        except Exception as e:
            print(f"❌ Face detector initialization failed: {e}")
            raise
    
    def calculate_single_video(self, video_path: str, verbose: bool = True) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate LSE metrics for a single video.
        
        Args:
            video_path: Path to the video file.
            verbose: Whether to print detailed information.
            
        Returns:
            A tuple of (lse_distance, lse_confidence).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if verbose:
            print(f"🎬 Calculating LSE for video: {os.path.basename(video_path)}")
            
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Step 1: Preprocess video
                if verbose:
                    print("  📝 Step 1: Preprocessing video...")
                    
                preprocessed_videos = self._preprocess_video(video_path, temp_dir, verbose)
                
                if not preprocessed_videos:
                    if verbose:
                        print("  ⚠️  Could not extract valid face segments.")
                    return None, None
                
                # Step 2: Calculate LSE scores
                if verbose:
                    print(f"  🧮 Step 2: Calculating LSE scores for {len(preprocessed_videos)} segments...")
                    
                distances, confidences = [], []
                
                for video_file in preprocessed_videos:
                    dist, conf = self._calculate_lse_for_clip(video_file, temp_dir, verbose)
                    if dist is not None and conf is not None:
                        distances.append(dist)
                        confidences.append(conf)
                
                if not distances:
                    if verbose:
                        print("  ⚠️  Could not calculate LSE scores.")
                    return None, None
                
                # Calculate average values
                avg_distance = np.mean(distances)
                avg_confidence = np.mean(confidences)
                
                elapsed = time.time() - start_time
                if verbose:
                    print(f"  ✅ LSE calculation completed in {elapsed:.2f}s")
                    print(f"     LSE Distance: {avg_distance:.4f}")
                    print(f"     LSE Confidence: {avg_confidence:.4f}")
                
                return avg_distance, avg_confidence
                
            except Exception as e:
                if verbose:
                    print(f"  ❌ LSE calculation failed: {e}")
                return None, None
    
    def calculate_batch(self, video_paths: List[str], verbose: bool = True) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Batch calculate LSE metrics for multiple videos.
        
        Args:
            video_paths: List of video file paths.
            verbose: Whether to print detailed information.
            
        Returns:
            A dictionary with video paths as keys and (lse_distance, lse_confidence) as values.
        """
        results = {}
        
        if verbose:
            print(f"🚀 Starting batch LSE calculation for {len(video_paths)} videos")
        
        for i, video_path in enumerate(video_paths, 1):
            if verbose:
                print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
            
            try:
                lse_d, lse_c = self.calculate_single_video(video_path, verbose=verbose)
                results[video_path] = (lse_d, lse_c)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Processing failed: {e}")
                results[video_path] = (None, None)
        
        if verbose:
            print(f"\n✅ Batch calculation complete")
            success_count = sum(1 for v in results.values() if v[0] is not None)
            print(f"   Success: {success_count}/{len(video_paths)}")
        
        return results
    
    def _preprocess_video(self, video_path: str, work_dir: str, verbose: bool = False) -> List[str]:
        """Preprocess video and extract face segments."""
        
        # Create working directories
        reference = "temp_video"
        avi_dir = os.path.join(work_dir, "pyavi", reference)
        frames_dir = os.path.join(work_dir, "pyframes", reference)
        crop_dir = os.path.join(work_dir, "pycrop", reference)
        work_subdir = os.path.join(work_dir, "pywork", reference)
        tmp_dir = os.path.join(work_dir, "pytmp", reference)
        
        os.makedirs(avi_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(crop_dir, exist_ok=True)
        os.makedirs(work_subdir, exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Convert video format and extract frames
        video_avi = os.path.join(avi_dir, "video.avi")
        audio_wav = os.path.join(avi_dir, "audio.wav")
        
        # Convert video
        cmd = f"ffmpeg -y -i {video_path} -qscale:v 2 -async 1 -r {self.frame_rate} {video_avi}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Extract frames
        cmd = f"ffmpeg -y -i {video_avi} -qscale:v 2 -threads 1 -f image2 {frames_dir}/%06d.jpg"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Extract audio
        cmd = f"ffmpeg -y -i {video_avi} -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_wav}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Face detection
        if verbose:
            print("    🔍 Detecting faces...")
        faces = self._detect_faces(frames_dir)
        
        # Scene detection
        if verbose:
            print("    🎬 Detecting scenes...")
        scenes = self._detect_scenes(video_avi)
        
        # Face tracking
        if verbose:
            print("    👤 Tracking faces...")
        tracks = []
        for scene in scenes:
            if scene[1].frame_num - scene[0].frame_num >= self.min_track:
                scene_faces = faces[scene[0].frame_num:scene[1].frame_num]
                tracks.extend(self._track_faces(scene_faces))
        
        # Crop face videos
        if verbose:
            print("    ✂️  Cropping face videos...")
        cropped_videos = []
        for ii, track in enumerate(tracks):
            output_path = os.path.join(crop_dir, f"{ii:05d}.avi")
            if self._crop_face_video(track, frames_dir, audio_wav, output_path):
                cropped_videos.append(output_path)
        
        return cropped_videos
    
    def _detect_faces(self, frames_dir: str) -> List[List[Dict]]:
        """Detect faces in all frames of a directory."""
        flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        
        dets = []
        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(image_rgb, conf_th=0.9, scales=[self.facedet_scale])
            
            frame_dets = []
            for bbox in bboxes:
                frame_dets.append({
                    'frame': fidx, 
                    'bbox': bbox[:-1].tolist(), 
                    'conf': bbox[-1]
                })
            dets.append(frame_dets)
        
        return dets
    
    def _detect_scenes(self, video_path: str) -> List[Tuple]:
        """Detect scenes in a video using the new PySceneDetect API."""
        try:
            # Use the new scenedetect.detect API
            scene_list = detect(video_path, ContentDetector())
            
            # If no scenes detected, treat the entire video as one scene
            if not scene_list:
                # Open video to get duration information
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        duration_seconds = frame_count / fps
                        start_time = FrameTimecode('00:00:00.000', fps=fps)
                        end_time = FrameTimecode(timecode=duration_seconds, fps=fps)
                        scene_list = [(start_time, end_time)]
                    cap.release()
                
                # Fallback if we can't get video info
                if not scene_list:
                    fps = 25.0  # Default FPS
                    start_time = FrameTimecode('00:00:00.000', fps=fps)
                    end_time = FrameTimecode('00:00:10.000', fps=fps)  # Assume 10 second video
                    scene_list = [(start_time, end_time)]
            
            return scene_list
            
        except Exception as e:
            print(f"⚠️ Scene detection failed: {e}")
            # Fallback: treat entire video as one scene
            fps = 25.0  # Default FPS
            start_time = FrameTimecode('00:00:00.000', fps=fps)
            end_time = FrameTimecode('00:00:10.000', fps=fps)  # Assume 10 second video
            return [(start_time, end_time)]
    
    def _track_faces(self, scene_faces: List[List[Dict]]) -> List[Dict]:
        """Track faces within a scene."""
        iou_threshold = 0.5
        tracks = []
        
        while True:
            track = []
            for frame_faces in scene_faces:
                for face in frame_faces:
                    if not track:
                        track.append(face)
                        frame_faces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:
                        iou = self._calculate_iou(face['bbox'], track[-1]['bbox'])
                        if iou > iou_threshold:
                            track.append(face)
                            frame_faces.remove(face)
                            continue
                    else:
                        break
            
            if not track:
                break
            elif len(track) > self.min_track:
                # Interpolate track
                framenum = np.array([f['frame'] for f in track])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                
                frame_i = np.arange(framenum[0], framenum[-1] + 1)
                
                bboxes_i = []
                for ij in range(4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)
                
                # Check face size
                face_width = np.mean(bboxes_i[:, 2] - bboxes_i[:, 0])
                face_height = np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])
                
                if max(face_width, face_height) > self.min_face_size:
                    tracks.append({'frame': frame_i, 'bbox': bboxes_i})
        
        return tracks
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    
    def _crop_face_video(self, track: Dict, frames_dir: str, audio_path: str, output_path: str) -> bool:
        """Crop a video to a tracked face."""
        try:
            flist = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            temp_video = output_path + "t.avi"
            vOut = cv2.VideoWriter(temp_video, fourcc, self.frame_rate, (224, 224))
            
            dets = {'x': [], 'y': [], 's': []}
            
            for det in track['bbox']:
                dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
                dets['y'].append((det[1] + det[3]) / 2)
                dets['x'].append((det[0] + det[2]) / 2)
            
            # Smooth detection results
            dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
            dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
            dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
            
            for fidx, frame in enumerate(track['frame']):
                cs = self.crop_scale
                bs = dets['s'][fidx]
                bsi = int(bs * (1 + 2 * cs))
                
                image = cv2.imread(flist[frame])
                frame_padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
                
                my = dets['y'][fidx] + bsi
                mx = dets['x'][fidx] + bsi
                
                face = frame_padded[int(my - bs):int(my + bs * (1 + 2 * cs)),
                                   int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
                
                vOut.write(cv2.resize(face, (224, 224)))
            
            vOut.release()
            
            # Crop audio
            audiostart = track['frame'][0] / self.frame_rate
            audioend = (track['frame'][-1] + 1) / self.frame_rate
            
            temp_audio = output_path.replace('.avi', '_audio.wav')
            cmd = f"ffmpeg -y -i {audio_path} -ss {audiostart:.3f} -to {audioend:.3f} {temp_audio}"
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Combine audio and video
            cmd = f"ffmpeg -y -i {temp_video} -i {temp_audio} -c:v copy -c:a copy {output_path}"
            result = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Clean up temporary files
            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return result == 0
            
        except Exception as e:
            print(f"Failed to crop video: {e}")
            return False
    
    def _calculate_lse_for_clip(self, video_path: str, work_dir: str, verbose: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """Calculate LSE for a single video clip."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract frames and audio
                frame_pattern = os.path.join(temp_dir, "%06d.jpg")
                audio_path = os.path.join(temp_dir, "audio.wav")
                
                # Extract frames
                cmd = f"ffmpeg -loglevel error -y -i {video_path} -threads 1 -f image2 {frame_pattern}"
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Extract audio
                cmd = f"ffmpeg -loglevel error -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}"
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Load video frames
                images = []
                flist = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
                
                for fname in flist:
                    img_input = cv2.imread(fname)
                    img_input = cv2.resize(img_input, (224, 224))
                    images.append(img_input)
                
                if len(images) < 5:
                    return None, None
                
                im = np.stack(images, axis=3)
                im = np.expand_dims(im, axis=0)
                im = np.transpose(im, (0, 3, 4, 1, 2))
                imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
                
                # Load audio
                if not os.path.exists(audio_path):
                    return None, None
                
                sample_rate, audio = wavfile.read(audio_path)
                mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
                mfcc = np.stack([np.array(i) for i in mfcc])
                
                cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
                cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
                
                # Check length
                min_length = min(len(images), len(audio) // 640)
                lastframe = min_length - 5
                
                if lastframe <= 0:
                    return None, None
                
                # Extract features
                im_feat = []
                cc_feat = []
                
                for i in range(0, lastframe, self.batch_size):
                    # Video features
                    im_batch = [imtv[:, :, vframe:vframe + 5, :, :] 
                               for vframe in range(i, min(lastframe, i + self.batch_size))]
                    im_in = torch.cat(im_batch, 0)
                    
                    if self.device == "cuda":
                        im_in = im_in.cuda()
                    
                    im_out = self.syncnet.forward_lip(im_in)
                    im_feat.append(im_out.data.cpu())
                    
                    # Audio features
                    cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] 
                               for vframe in range(i, min(lastframe, i + self.batch_size))]
                    cc_in = torch.cat(cc_batch, 0)
                    
                    if self.device == "cuda":
                        cc_in = cc_in.cuda()
                    
                    cc_out = self.syncnet.forward_aud(cc_in)
                    cc_feat.append(cc_out.data.cpu())
                
                im_feat = torch.cat(im_feat, 0)
                cc_feat = torch.cat(cc_feat, 0)
                
                # Calculate distances
                dists = self._calc_pdist(im_feat, cc_feat, vshift=self.vshift)
                mdist = torch.mean(torch.stack(dists, 1), 1)
                
                minval, minidx = torch.min(mdist, 0)
                conf = torch.median(mdist) - minval
                
                return minval.numpy().item(), conf.numpy().item()
                
        except Exception as e:
            if verbose:
                print(f"Failed to calculate LSE: {e}")
            return None, None
    
    def _calc_pdist(self, feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 10) -> List[torch.Tensor]:
        """Calculate pairwise distance between features."""
        win_size = vshift * 2 + 1
        feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
        
        dists = []
        for i in range(len(feat1)):
            dists.append(torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), 
                feat2p[i:i + win_size, :]
            ))
        
        return dists


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LSE Calculator")
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Computation device")
    
    args = parser.parse_args()
    
    # Create calculator instance
    calculator = LSECalculator(device=args.device)
    
    # Calculate LSE
    lse_d, lse_c = calculator.calculate_single_video(args.video, verbose=True)
    
    if lse_d is not None and lse_c is not None:
        print(f"\n📊 Final Results:")
        print(f"   LSE Distance: {lse_d:.4f}")
        print(f"   LSE Confidence: {lse_c:.4f}")
    else:
        print(f"\n❌ LSE calculation failed.")


if __name__ == "__main__":
    main()