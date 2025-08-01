#!/usr/bin/env python3
"""
LSE (Lip-Sync Error) Calculator
基于SyncNet的Python API，计算LSE-D (Distance) 和 LSE-C (Confidence) 指标

Copyright (c) 2025 Fating Hong <fatinghong@gmail.com>
All rights reserved.

This module provides Python API for SyncNet-based lip-sync error calculation.

使用方法:
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
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

# 导入本地的SyncNet模块
try:
    # 作为包导入时使用相对导入
    from .syncnet_core.model import S
    from .syncnet_core.detectors import S3FD
except ImportError:
    # 直接运行时使用绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from syncnet_core.model import S
    from syncnet_core.detectors import S3FD


class LSECalculator:
    """LSE计算器"""
    
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
        初始化LSE计算器
        
        Args:
            model_path: SyncNet模型路径，如果为None则使用默认路径
            device: 计算设备 ("cuda" 或 "cpu")
            batch_size: 批处理大小
            vshift: 视频偏移范围
            facedet_scale: 人脸检测缩放因子
            crop_scale: 裁剪缩放因子
            min_track: 最小跟踪持续时间
            frame_rate: 帧率
            num_failed_det: 允许的检测失败次数
            min_face_size: 最小人脸大小(像素)
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
        
        # 确定模型路径
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "models", "syncnet_v2.model")
        
        # 初始化SyncNet模型
        print(f"🔄 正在加载SyncNet模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SyncNet模型文件不存在: {model_path}")
            
        self.syncnet = S(num_layers_in_fc_layers=1024)
        if self.device == "cuda" and torch.cuda.is_available():
            self.syncnet = self.syncnet.cuda()
        else:
            self.device = "cpu"
            
        # 加载模型参数
        loaded_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        self_state = self.syncnet.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
        self.syncnet.eval()
        
        # 初始化人脸检测器
        print(f"🔄 正在初始化人脸检测器")
        try:
            self.face_detector = S3FD(device=self.device)
            print(f"✅ LSE计算器初始化完成 (设备: {self.device})")
        except Exception as e:
            print(f"❌ 人脸检测器初始化失败: {e}")
            raise
    
    def calculate_single_video(self, video_path: str, verbose: bool = True) -> Tuple[Optional[float], Optional[float]]:
        """
        计算单个视频的LSE指标
        
        Args:
            video_path: 视频文件路径
            verbose: 是否打印详细信息
            
        Returns:
            (lse_distance, lse_confidence): LSE距离和置信度
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        if verbose:
            print(f"🎬 计算视频LSE: {os.path.basename(video_path)}")
            
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 步骤1: 预处理视频
                if verbose:
                    print("  📝 步骤1: 预处理视频...")
                    
                preprocessed_videos = self._preprocess_video(video_path, temp_dir, verbose)
                
                if not preprocessed_videos:
                    if verbose:
                        print("  ⚠️  无法提取有效的人脸片段")
                    return None, None
                
                # 步骤2: 计算LSE分数
                if verbose:
                    print(f"  🧮 步骤2: 计算LSE分数 ({len(preprocessed_videos)}个片段)...")
                    
                distances, confidences = [], []
                
                for video_file in preprocessed_videos:
                    dist, conf = self._calculate_lse_for_clip(video_file, temp_dir, verbose)
                    if dist is not None and conf is not None:
                        distances.append(dist)
                        confidences.append(conf)
                
                if not distances:
                    if verbose:
                        print("  ⚠️  无法计算LSE分数")
                    return None, None
                
                # 计算平均值
                avg_distance = np.mean(distances)
                avg_confidence = np.mean(confidences)
                
                elapsed = time.time() - start_time
                if verbose:
                    print(f"  ✅ LSE计算完成 ({elapsed:.2f}s)")
                    print(f"     LSE距离: {avg_distance:.4f}")
                    print(f"     LSE置信度: {avg_confidence:.4f}")
                
                return avg_distance, avg_confidence
                
            except Exception as e:
                if verbose:
                    print(f"  ❌ LSE计算失败: {e}")
                return None, None
    
    def calculate_batch(self, video_paths: List[str], verbose: bool = True) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        批量计算多个视频的LSE指标
        
        Args:
            video_paths: 视频文件路径列表
            verbose: 是否打印详细信息
            
        Returns:
            字典，键为视频路径，值为(lse_distance, lse_confidence)
        """
        results = {}
        
        if verbose:
            print(f"🚀 开始批量LSE计算 ({len(video_paths)}个视频)")
        
        for i, video_path in enumerate(video_paths, 1):
            if verbose:
                print(f"\n[{i}/{len(video_paths)}] 处理: {os.path.basename(video_path)}")
            
            try:
                lse_d, lse_c = self.calculate_single_video(video_path, verbose=verbose)
                results[video_path] = (lse_d, lse_c)
            except Exception as e:
                if verbose:
                    print(f"  ❌ 处理失败: {e}")
                results[video_path] = (None, None)
        
        if verbose:
            print(f"\n✅ 批量计算完成")
            success_count = sum(1 for v in results.values() if v[0] is not None)
            print(f"   成功: {success_count}/{len(video_paths)}")
        
        return results
    
    def _preprocess_video(self, video_path: str, work_dir: str, verbose: bool = False) -> List[str]:
        """预处理视频，提取人脸片段"""
        
        # 创建工作目录
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
        
        # 转换视频格式和提取帧
        video_avi = os.path.join(avi_dir, "video.avi")
        audio_wav = os.path.join(avi_dir, "audio.wav")
        
        # 转换视频
        cmd = f"ffmpeg -y -i {video_path} -qscale:v 2 -async 1 -r {self.frame_rate} {video_avi}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 提取帧
        cmd = f"ffmpeg -y -i {video_avi} -qscale:v 2 -threads 1 -f image2 {frames_dir}/%06d.jpg"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 提取音频
        cmd = f"ffmpeg -y -i {video_avi} -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_wav}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 人脸检测
        if verbose:
            print("    🔍 检测人脸...")
        faces = self._detect_faces(frames_dir)
        
        # 场景检测
        if verbose:
            print("    🎬 检测场景...")
        scenes = self._detect_scenes(video_avi)
        
        # 人脸跟踪
        if verbose:
            print("    👤 跟踪人脸...")
        tracks = []
        for scene in scenes:
            if scene[1].frame_num - scene[0].frame_num >= self.min_track:
                scene_faces = faces[scene[0].frame_num:scene[1].frame_num]
                tracks.extend(self._track_faces(scene_faces))
        
        # 裁剪人脸视频
        if verbose:
            print("    ✂️  裁剪人脸视频...")
        cropped_videos = []
        for ii, track in enumerate(tracks):
            output_path = os.path.join(crop_dir, f"{ii:05d}.avi")
            if self._crop_face_video(track, frames_dir, audio_wav, output_path):
                cropped_videos.append(output_path)
        
        return cropped_videos
    
    def _detect_faces(self, frames_dir: str) -> List[List[Dict]]:
        """检测人脸"""
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
        """检测场景"""
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()
        
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        
        if not scene_list:
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
        
        return scene_list
    
    def _track_faces(self, scene_faces: List[List[Dict]]) -> List[Dict]:
        """跟踪人脸"""
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
                # 插值轨迹
                framenum = np.array([f['frame'] for f in track])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                
                frame_i = np.arange(framenum[0], framenum[-1] + 1)
                
                bboxes_i = []
                for ij in range(4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)
                
                # 检查人脸大小
                face_width = np.mean(bboxes_i[:, 2] - bboxes_i[:, 0])
                face_height = np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])
                
                if max(face_width, face_height) > self.min_face_size:
                    tracks.append({'frame': frame_i, 'bbox': bboxes_i})
        
        return tracks
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
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
        """裁剪人脸视频"""
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
            
            # 平滑检测结果
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
            
            # 裁剪音频
            audiostart = track['frame'][0] / self.frame_rate
            audioend = (track['frame'][-1] + 1) / self.frame_rate
            
            temp_audio = output_path.replace('.avi', '_audio.wav')
            cmd = f"ffmpeg -y -i {audio_path} -ss {audiostart:.3f} -to {audioend:.3f} {temp_audio}"
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 合并音视频
            cmd = f"ffmpeg -y -i {temp_video} -i {temp_audio} -c:v copy -c:a copy {output_path}"
            result = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 清理临时文件
            if os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return result == 0
            
        except Exception as e:
            print(f"裁剪视频失败: {e}")
            return False
    
    def _calculate_lse_for_clip(self, video_path: str, work_dir: str, verbose: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """计算单个视频片段的LSE"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 提取帧和音频
                frame_pattern = os.path.join(temp_dir, "%06d.jpg")
                audio_path = os.path.join(temp_dir, "audio.wav")
                
                # 提取帧
                cmd = f"ffmpeg -loglevel error -y -i {video_path} -threads 1 -f image2 {frame_pattern}"
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 提取音频
                cmd = f"ffmpeg -loglevel error -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}"
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 加载视频帧
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
                
                # 加载音频
                if not os.path.exists(audio_path):
                    return None, None
                
                sample_rate, audio = wavfile.read(audio_path)
                mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
                mfcc = np.stack([np.array(i) for i in mfcc])
                
                cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
                cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
                
                # 检查长度
                min_length = min(len(images), len(audio) // 640)
                lastframe = min_length - 5
                
                if lastframe <= 0:
                    return None, None
                
                # 提取特征
                im_feat = []
                cc_feat = []
                
                for i in range(0, lastframe, self.batch_size):
                    # 视频特征
                    im_batch = [imtv[:, :, vframe:vframe + 5, :, :] 
                               for vframe in range(i, min(lastframe, i + self.batch_size))]
                    im_in = torch.cat(im_batch, 0)
                    
                    if self.device == "cuda":
                        im_in = im_in.cuda()
                    
                    im_out = self.syncnet.forward_lip(im_in)
                    im_feat.append(im_out.data.cpu())
                    
                    # 音频特征
                    cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] 
                               for vframe in range(i, min(lastframe, i + self.batch_size))]
                    cc_in = torch.cat(cc_batch, 0)
                    
                    if self.device == "cuda":
                        cc_in = cc_in.cuda()
                    
                    cc_out = self.syncnet.forward_aud(cc_in)
                    cc_feat.append(cc_out.data.cpu())
                
                im_feat = torch.cat(im_feat, 0)
                cc_feat = torch.cat(cc_feat, 0)
                
                # 计算距离
                dists = self._calc_pdist(im_feat, cc_feat, vshift=self.vshift)
                mdist = torch.mean(torch.stack(dists, 1), 1)
                
                minval, minidx = torch.min(mdist, 0)
                conf = torch.median(mdist) - minval
                
                return minval.numpy().item(), conf.numpy().item()
                
        except Exception as e:
            if verbose:
                print(f"计算LSE失败: {e}")
            return None, None
    
    def _calc_pdist(self, feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 10) -> List[torch.Tensor]:
        """计算特征距离"""
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
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LSE计算器")
    parser.add_argument("--video", type=str, required=True, help="输入视频文件")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    
    args = parser.parse_args()
    
    # 创建计算器
    calculator = LSECalculator(device=args.device)
    
    # 计算LSE
    lse_d, lse_c = calculator.calculate_single_video(args.video, verbose=True)
    
    if lse_d is not None and lse_c is not None:
        print(f"\n📊 最终结果:")
        print(f"   LSE Distance: {lse_d:.4f}")
        print(f"   LSE Confidence: {lse_c:.4f}")
    else:
        print(f"\n❌ LSE计算失败")


if __name__ == "__main__":
    main() 