"""
YouTube Downloader Pro + Video Clipper - Enhanced Version
Author: Enhanced by AI
Description: Multi-threaded YouTube downloader with advanced video clipping features
Version: 2.0 Ultra Quality
"""

import yt_dlp
import os
import json
import threading
import subprocess
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import socket
import urllib.parse
import traceback
import re

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_PATH = r"C:\Users\BAGUS\ai-podcast-clipper\youtube_download"

DEFAULT_CONFIG = {
    # Download settings - MAXIMUM QUALITY
    'output_template': '%(title)s.%(ext)s',
    'max_concurrent_downloads': 1,
    'default_format': 'best[height<=1080]',
    'fallback_format': 'best[height<=720]',
    'retry_count': 5,
    'max_retries': 10,
    'timeout': 60,
    'ignore_errors': True,
    'resume_downloads': True,
    'validate_urls': True,
    'suppress_warnings': True,
    'socket_timeout': 30,
    'max_downloads_per_host': 1,
    'throttled_rate': '10M',
    
    # Media settings
    'download_subtitles': False,
    'subtitles_langs': ['en'],
    'embed_metadata': True,
    'embed_thumbnail': True,
    'embed_subs': False,
    'keep_subs': False,
    'use_android_client': False,
    
    # Clipping settings - PRO QUALITY
    'clipping_preset': 'high_quality',
    'keep_original': True,
    'auto_detect_silence': False,
    'min_clip_duration': 10,
    'max_clip_duration': 600,
    'audio_bitrate': '320k',
    'audio_sample_rate': '48000',
    'video_crf': '17',
    'video_preset': 'slow',
    'video_codec': 'libx264',
    'audio_codec': 'aac',
    'pixel_format': 'yuv420p',
    'profile': 'high',
    'level': '4.2',
    'keyframe_interval': 30,
    'scene_cut': 40,
    'tune': 'film',
    'deblock': '1:1',
    
    # Directory settings
    'clips_output_dir': 'clips',
    'completed_dir': 'completed',
    'temp_dir': 'temp',
    'move_to_completed': False,
    'debug_mode': False,
    'log_level': 'INFO',
    
    # Advanced settings
    'ffmpeg_location': None,
    'ffprobe_location': None,
    'hwaccel': 'auto',
    'hwaccel_device': None,
    'use_hardware_decoding': False,
    'use_hardware_encoding': False,
    'multithread_encoding': True,
    'thread_count': 0,
}


# ==================== DATA CLASSES ====================
@dataclass
class ClipSegment:
    """Data class untuk segment video"""
    start_time: str
    end_time: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    output_filename: str = ""
    quality_preset: str = "high_quality"
    
    def __post_init__(self):
        if not self.output_filename and self.title:
            safe_title = re.sub(r'[^\w\s-]', '', self.title).strip().replace(' ', '_')
            self.output_filename = f"{safe_title}.mp4"


@dataclass
class AudioExtractConfig:
    """Data class untuk konfigurasi ekstraksi audio"""
    start_time: str
    end_time: str
    output_format: str = 'mp3'
    bitrate: str = '320k'
    sample_rate: str = '48000'
    channels: int = 2
    remove_noise: bool = False
    normalize_audio: bool = True
    audio_filter: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class VideoQualityProfile:
    """Profile kualitas video"""
    name: str
    crf: str
    preset: str
    video_bitrate: str = None
    audio_bitrate: str = '320k'
    tune: str = 'film'
    profile: str = 'high'
    level: str = '4.2'
    description: str = ""


# ==================== THREAD-SAFE PRINTER ====================
class ThreadSafePrinter:
    """Thread-safe printer untuk output yang teratur"""
    
    def __init__(self):
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._printer_thread = threading.Thread(target=self._worker, daemon=True)
        self._printer_thread.start()
    
    def _worker(self):
        """Worker thread untuk printing"""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=1)
                if item:
                    print(*item if isinstance(item, tuple) else (item,), flush=True)
                self._queue.task_done()
            except:
                pass
    
    def print(self, *args):
        self._queue.put(args)
    
    def stop(self):
        self._stop_event.set()
        self._printer_thread.join(timeout=2)


# ==================== VIDEO PROCESSOR ====================
class VideoProcessor:
    """Class untuk memproses video: clipping, trimming, merging"""
    
    QUALITY_PROFILES = {
        'ultra_fast': VideoQualityProfile(
            name='ultra_fast',
            crf='28',
            preset='ultrafast',
            audio_bitrate='128k',
            tune='fastdecode',
            profile='main',
            description="Sangat cepat, kualitas rendah"
        ),
        'fast': VideoQualityProfile(
            name='fast',
            crf='26',
            preset='veryfast',
            audio_bitrate='192k',
            profile='main',
            description="Cepat, kualitas medium"
        ),
        'medium': VideoQualityProfile(
            name='medium',
            crf='23',
            preset='medium',
            audio_bitrate='256k',
            description="Seimbang"
        ),
        'high_quality': VideoQualityProfile(
            name='high_quality',
            crf='18',
            preset='slow',
            audio_bitrate='320k',
            tune='film',
            profile='high',
            level='4.2',
            description="Kualitas tinggi, kompresi optimal"
        ),
        'pro': VideoQualityProfile(
            name='pro',
            crf='16',
            preset='slower',
            video_bitrate='10000k',
            audio_bitrate='320k',
            tune='film',
            profile='high',
            level='5.0',
            description="Kualitas profesional, file besar"
        ),
        'lossless': VideoQualityProfile(
            name='lossless',
            crf='0',
            preset='veryslow',
            tune='film',
            profile='high444',
            description="Hampir lossless, file sangat besar"
        ),
        'youtube': VideoQualityProfile(
            name='youtube',
            crf='20',
            preset='medium',
            audio_bitrate='256k',
            tune='film',
            profile='high',
            level='4.2',
            description="Optimal untuk upload YouTube"
        )
    }
    
    def __init__(self, base_path: str, config: dict):
        self.base_path = Path(base_path)
        self.config = config or {}
        
        # Inisialisasi hwaccel_device dari config
        self.hwaccel = self.config.get('hwaccel', 'auto')
        self.hwaccel_device = self.config.get('hwaccel_device')
        
        clips_dir_name = self.config.get('clips_output_dir', 'clips')
        completed_dir_name = self.config.get('completed_dir', 'completed')
        temp_dir_name = self.config.get('temp_dir', 'temp_clips')
        
        self.clips_dir = self.base_path / clips_dir_name
        self.completed_dir = self.base_path / completed_dir_name
        self.temp_dir = self.base_path / temp_dir_name
        
        for directory in [self.clips_dir, self.completed_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
        
        self.ffmpeg_path = self.config.get('ffmpeg_location')
        self.ffprobe_path = self.config.get('ffprobe_location')
        self.ffmpeg_available = self._check_ffmpeg()
        
        preset_name = self.config.get('clipping_preset', 'high_quality')
        self.quality_profile = self._get_quality_profile(preset_name)
        
    def _check_ffmpeg(self) -> bool:
        """Cek ketersediaan ffmpeg"""
        try:
            cmd = ['ffmpeg', '-version'] if not self.ffmpeg_path else [self.ffmpeg_path, '-version']
            result = subprocess.run(
                cmd,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_quality_profile(self, preset_name: str) -> VideoQualityProfile:
        """Get quality profile berdasarkan preset"""
        return self.QUALITY_PROFILES.get(preset_name, self.QUALITY_PROFILES['high_quality'])
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Konversi string waktu ke detik"""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(time_str)
    
    def _seconds_to_time(self, seconds: float) -> str:
        """Konversi detik ke HH:MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Dapatkan informasi video menggunakan ffprobe"""
        try:
            cmd = ['ffprobe'] if not self.ffprobe_path else [self.ffprobe_path]
            cmd.extend([
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                video_path
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            
            return {
                'duration': float(data['format']['duration']),
                'bitrate': data['format'].get('bit_rate'),
                'size': data['format'].get('size'),
                'format_name': data['format'].get('format_name', ''),
                'video_codec': video_stream.get('codec_name') if video_stream else None,
                'video_width': video_stream.get('width') if video_stream else None,
                'video_height': video_stream.get('height') if video_stream else None,
                'video_bitrate': video_stream.get('bit_rate') if video_stream else None,
                'video_fps': eval(video_stream.get('avg_frame_rate', '0/1')) if video_stream else None,
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'audio_bitrate': audio_stream.get('bit_rate') if audio_stream else None,
                'audio_sample_rate': audio_stream.get('sample_rate') if audio_stream else None,
            }
        except Exception as e:
            print(f"⚠️  Gagal get video info: {e}")
            return {}
    
    def get_video_duration(self, video_path: str) -> str:
        """Dapatkan durasi video dalam format readable"""
        info = self.get_video_info(video_path)
        duration = info.get('duration', 0)
        return self._seconds_to_time(duration)
    
    def clip_video(self,
                  input_path: str,
                  segments: List[ClipSegment],
                  output_dir: Optional[str] = None,
                  preserve_quality: bool = True) -> List[str]:
        """
        Potong video menjadi beberapa segmen dengan kualitas maksimal
        """
        if not self.ffmpeg_available:
            print("❌ FFmpeg tidak tersedia!")
            return []
        
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"❌ File tidak ditemukan: {input_path}")
            return []
        
        output_dir = Path(output_dir) if output_dir else self.clips_dir
        output_dir.mkdir(exist_ok=True)
        
        video_info = self.get_video_info(str(input_path))
        total_duration = video_info.get('duration', 0)
        
        if not video_info:
            print("❌ Tidak bisa membaca video info")
            return []
        
        output_files = []
        
        for i, segment in enumerate(segments):
            try:
                start_sec = max(0, self._time_to_seconds(segment.start_time))
                end_sec = min(self._time_to_seconds(segment.end_time), total_duration)
                
                if start_sec >= end_sec:
                    print(f"⚠️  Segment {i+1}: Waktu tidak valid, dilewati")
                    continue
                
                duration = end_sec - start_sec
                if duration < 1:
                    print(f"⚠️  Segment {i+1}: Durasi terlalu pendek, dilewati")
                    continue
                
                if not segment.output_filename:
                    safe_name = re.sub(r'[^\w\s-]', '', input_path.stem).strip().replace(' ', '_')
                    segment.output_filename = f"clip_{i+1:03d}_{safe_name}.mp4"
                
                output_path = output_dir / segment.output_filename
                
                counter = 1
                while output_path.exists():
                    output_path = output_path.parent / f"{output_path.stem}_{counter:03d}{output_path.suffix}"
                    counter += 1
                
                print(f"\n✂️  Segment {i+1}: {self._seconds_to_time(start_sec)} - {self._seconds_to_time(end_sec)}")
                print(f"   Durasi: {self._seconds_to_time(duration)}")
                print(f"   Output: {output_path.name}")
                
                segment_profile = self._get_quality_profile(segment.quality_preset)
                
                ffmpeg_cmd = [self.ffmpeg_path] if self.ffmpeg_path else ['ffmpeg']
                
                # PERBAIKAN: Hanya gunakan hwaccel jika tidak 'none'
                if self.hwaccel and self.hwaccel.lower() != 'none':
                    ffmpeg_cmd.extend(['-hwaccel', self.hwaccel])
                    # Hanya tambahkan hwaccel_device jika tidak None
                    if self.hwaccel_device:
                        ffmpeg_cmd.extend(['-hwaccel_device', self.hwaccel_device])
                
                ffmpeg_cmd.extend([
                    '-ss', str(start_sec),
                    '-i', str(input_path),
                    '-t', str(duration),
                    '-map_metadata', '0',
                    '-map_chapters', '0',
                ])
                
                if segment.quality_preset == 'fast' and preserve_quality:
                    ffmpeg_cmd.extend([
                        '-c:v', 'copy',
                        '-c:a', 'copy',
                    ])
                else:
                    ffmpeg_cmd.extend([
                        '-c:v', self.config.get('video_codec', 'libx264'),
                        '-crf', segment_profile.crf,
                        '-preset', segment_profile.preset,
                        '-tune', segment_profile.tune,
                        '-profile:v', segment_profile.profile,
                        '-level:v', segment_profile.level,
                        '-pix_fmt', self.config.get('pixel_format', 'yuv420p'),
                        '-g', str(self.config.get('keyframe_interval', 30)),
                        '-sc_threshold', str(self.config.get('scene_cut', 40)),
                        '-deblock', self.config.get('deblock', '1:1'),
                    ])
                    
                    if segment_profile.video_bitrate:
                        ffmpeg_cmd.extend(['-b:v', segment_profile.video_bitrate])
                        ffmpeg_cmd.extend(['-maxrate', segment_profile.video_bitrate])
                    
                    if self.config.get('multithread_encoding', True):
                        thread_count = self.config.get('thread_count', 0)
                        ffmpeg_cmd.extend(['-threads', str(thread_count)])
                    
                    ffmpeg_cmd.extend([
                        '-c:a', self.config.get('audio_codec', 'aac'),
                        '-b:a', segment_profile.audio_bitrate,
                        '-ar', self.config.get('audio_sample_rate', '48000'),
                        '-ac', '2',
                    ])
                    
                    if segment.title:
                        ffmpeg_cmd.extend(['-metadata', f'title={segment.title}'])
                
                ffmpeg_cmd.extend([
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    str(output_path)
                ])
                
                print(f"   Encoding dengan preset: {segment_profile.name}")
                
                try:
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    
                    if process.returncode != 0:
                        print(f"❌ FFmpeg error: {process.stderr[:200]}")
                        continue
                    
                    if output_path.exists():
                        output_info = self.get_video_info(str(output_path))
                        if output_info:
                            size_mb = int(output_info.get('size', 0)) / (1024 * 1024)
                            duration_out = output_info.get('duration', 0)
                            print(f"✅ Segment {i+1} selesai:")
                            print(f"   Size: {size_mb:.1f}MB | Durasi: {self._seconds_to_time(duration_out)}")
                            output_files.append(str(output_path))
                        else:
                            print(f"⚠️  Segment {i+1}: Output tidak valid")
                    else:
                        print(f"❌ Segment {i+1}: Output file tidak dibuat")
                        
                except Exception as e:
                    print(f"❌ Gagal potong segment {i+1}: {e}")
                    continue
                
            except Exception as e:
                print(f"❌ Error pada segment {i+1}: {e}")
                continue
        
        print(f"\n🎬 Total {len(output_files)} segment berhasil diproses")
        return output_files
    
    def auto_detect_segments(self,
                           video_path: str,
                           min_duration: int = 10,
                           max_duration: int = 300,
                           method: str = "equal") -> List[ClipSegment]:
        """Auto-detect segments dengan berbagai metode"""
        print("🔍 Mendeteksi segment otomatis...")
        
        video_info = self.get_video_info(video_path)
        total_duration = video_info.get('duration', 0)
        
        if total_duration == 0:
            print("❌ Tidak bisa membaca durasi video")
            return []
        
        segments = []
        
        if method == "equal":
            segment_duration = min(max_duration, max(min_duration, total_duration / 5))
            
            start = 0
            segment_num = 1
            
            while start < total_duration:
                end = min(start + segment_duration, total_duration)
                
                if end - start >= min_duration:
                    segments.append(ClipSegment(
                        start_time=self._seconds_to_time(start),
                        end_time=self._seconds_to_time(end),
                        title=f"Segment {segment_num}",
                        output_filename=f"auto_segment_{segment_num:03d}.mp4"
                    ))
                    segment_num += 1
                
                start = end
        
        print(f"✅ Terdeteksi {len(segments)} segment")
        return segments
    
    def detect_silence(self,
                      video_path: str,
                      silence_threshold: float = -30.0,
                      min_silence_duration: float = 1.0) -> List[Tuple[float, float]]:
        """Deteksi periode silence dalam video"""
        try:
            ffmpeg_cmd = [self.ffmpeg_path] if self.ffmpeg_path else ['ffmpeg']
            ffmpeg_cmd.extend([
                '-i', video_path,
                '-af', f'silencedetect=noise={silence_threshold}dB:d={min_silence_duration}',
                '-f', 'null', '-'
            ])
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            silence_periods = []
            lines = result.stderr.split('\n')
            start_time = None
            
            for line in lines:
                if 'silence_start' in line:
                    match = re.search(r'silence_start: ([\d.]+)', line)
                    if match:
                        start_time = float(match.group(1))
                elif 'silence_end' in line and start_time is not None:
                    match = re.search(r'silence_end: ([\d.]+)', line)
                    if match:
                        end_time = float(match.group(1))
                        silence_periods.append((start_time, end_time))
                        start_time = None
            
            return silence_periods
            
        except Exception as e:
            print(f"⚠️  Gagal deteksi silence: {e}")
            return []
    
    def merge_clips(self,
                   input_paths: List[str],
                   output_path: str,
                   transition_effect: str = None) -> str:
        """Gabungkan beberapa klip dengan kualitas tinggi"""
        if not self.ffmpeg_available:
            print("❌ FFmpeg tidak tersedia!")
            return ""
        
        if len(input_paths) < 2:
            print("❌ Butuh minimal 2 video")
            return ""
        
        for path in input_paths:
            if not Path(path).exists():
                print(f"❌ File tidak ditemukan: {path}")
                return ""
        
        list_file = self.temp_dir / "concat_list.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for path in input_paths:
                f.write(f"file '{Path(path).absolute()}'\n")
        
        try:
            ffmpeg_cmd = [self.ffmpeg_path] if self.ffmpeg_path else ['ffmpeg']
            ffmpeg_cmd.extend([
                '-f', 'concat', '-safe', '0',
                '-i', str(list_file),
                '-c:v', self.config.get('video_codec', 'libx264'),
                '-crf', self.quality_profile.crf,
                '-preset', self.quality_profile.preset,
                '-c:a', self.config.get('audio_codec', 'aac'),
                '-b:a', self.quality_profile.audio_bitrate,
                '-movflags', '+faststart',
                '-y',
                output_path
            ])
            
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            print(f"✅ Berhasil menggabungkan {len(input_paths)} klip")
            
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"📁 Output size: {size_mb:.1f}MB")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Gagal merge: {e}")
            return ""
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""
        finally:
            if list_file.exists():
                list_file.unlink()
    
    def extract_audio_segment(self,
                            video_path: str,
                            config: AudioExtractConfig) -> str:
        """Ekstrak segmen audio dari video dengan kualitas tinggi"""
        if not self.ffmpeg_available:
            print("❌ FFmpeg tidak tersedia!")
            return ""
        
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ File tidak ditemukan: {video_path}")
            return ""
        
        start_sec = self._time_to_seconds(config.start_time)
        end_sec = self._time_to_seconds(config.end_time)
        duration = end_sec - start_sec
        
        if duration <= 0:
            print("❌ Durasi tidak valid")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_path.stem}_{timestamp}_audio.{config.output_format}"
        output_path = self.clips_dir / output_filename
        
        try:
            ffmpeg_cmd = [self.ffmpeg_path] if self.ffmpeg_path else ['ffmpeg']
            ffmpeg_cmd.extend([
                '-ss', str(start_sec),
                '-i', str(video_path),
                '-t', str(duration),
                '-vn',
                '-ar', config.sample_rate,
                '-ac', str(config.channels),
                '-b:a', config.bitrate,
            ])
            
            filters = []
            if config.normalize_audio:
                filters.append('loudnorm=I=-16:LRA=11:TP=-1.5')
            if config.remove_noise:
                filters.append('afftdn=nf=-25')
            if config.audio_filter:
                filters.append(config.audio_filter)
            
            if filters:
                ffmpeg_cmd.extend(['-af', ','.join(filters)])
            
            if config.output_format == 'mp3':
                ffmpeg_cmd.extend(['-q:a', '0'])
            elif config.output_format == 'flac':
                ffmpeg_cmd.extend(['-compression_level', '12'])
            elif config.output_format == 'wav':
                ffmpeg_cmd.extend(['-codec:a', 'pcm_s24le'])
            
            for key, value in config.metadata.items():
                ffmpeg_cmd.extend(['-metadata', f'{key}={value}'])
            
            ffmpeg_cmd.append(str(output_path))
            
            print(f"🎵 Ekstrak audio: {self._seconds_to_time(start_sec)} - {self._seconds_to_time(end_sec)}")
            
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Audio diekstrak: {output_path.name} ({size_mb:.1f}MB)")
                return str(output_path)
            else:
                print("❌ Gagal membuat file audio")
                return ""
            
        except Exception as e:
            print(f"❌ Gagal ekstrak audio: {e}")
            return ""
    
    def list_video_files(self, directory: Optional[str] = None) -> List[str]:
        """List semua file video di direktori dengan info detail"""
        directory = Path(directory) if directory else self.base_path
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv', '.m4v']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(directory.glob(f'*{ext}'))
            video_files.extend(directory.glob(f'*{ext.upper()}'))
        
        video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return [str(f) for f in video_files if f.is_file()]
    
    def get_video_details(self, video_path: str) -> Dict[str, Any]:
        """Get detailed video information"""
        info = self.get_video_info(video_path)
        if not info:
            return {}
        
        path = Path(video_path)
        return {
            'filename': path.name,
            'path': str(path),
            'size_mb': path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(path.stat().st_mtime),
            'duration': info.get('duration', 0),
            'resolution': f"{info.get('video_width', 0)}x{info.get('video_height', 0)}",
            'video_codec': info.get('video_codec', 'Unknown'),
            'video_bitrate': f"{int(info.get('video_bitrate', 0))//1000 if info.get('video_bitrate') else '?'}Kbps",
            'audio_codec': info.get('audio_codec', 'Unknown'),
            'fps': info.get('video_fps', 0),
        }
    
    def cleanup_temp_files(self):
        """Bersihkan file temporary"""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                print(f"🧹 Cleanup temporary files: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Gagal cleanup: {e}")


# ==================== YOUTUBE DOWNLOADER ====================
class YouTubeDownloader:
    """Main downloader class dengan kualitas maksimal"""
    
    def __init__(self):
        self.base_path = BASE_PATH
        
        self.config = self._load_initial_config()
        
        self._setup_paths()
        
        self._setup_logging()
        
        self.printer = ThreadSafePrinter()
        self.current_downloads = {}
        self.running = True
        self.js_runtime = self._check_js_runtime()
        self.video_processor = VideoProcessor(self.base_path, self.config)

    def safe_print(self, *args):
        """Thread-safe print"""
        if hasattr(self, 'printer'):
            self.printer.print(*args)
        else:
            print(*args)
        
    def _load_initial_config(self):
        """Load konfigurasi awal (tanpa logging)"""
        try:
            config_file = os.path.join(BASE_PATH, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    config = {**DEFAULT_CONFIG, **user_config}
            else:
                config = DEFAULT_CONFIG.copy()
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path and not config.get('ffmpeg_location'):
                config['ffmpeg_location'] = ffmpeg_path
                
            return config
            
        except Exception as e:
            print(f"⚠️  Config error: {e}")
            return DEFAULT_CONFIG
    
    def _setup_paths(self):
        """Setup semua path"""
        os.makedirs(self.base_path, exist_ok=True)
        
        self.history_file = os.path.join(self.base_path, 'download_history.json')
        self.config_file = os.path.join(self.base_path, 'config.json')
        self.status_file = os.path.join(self.base_path, 'download_status.json')
        self.downloads_dir = os.path.join(self.base_path, 'downloads')
        self.urls_file = os.path.join(self.base_path, 'urls.txt')
        self.temp_dir = os.path.join(self.base_path, 'temp')
        self.logs_dir = os.path.join(self.base_path, 'logs')
        
        for directory in [self.downloads_dir, self.temp_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging dengan config yang sudah diload"""
        import logging
        
        log_level = self.config.get('log_level', 'INFO')
        try:
            log_level = getattr(logging, log_level.upper())
        except AttributeError:
            log_level = logging.INFO
        
        logging.basicConfig(
            filename=os.path.join(self.logs_dir, 'downloader.log'),
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger = logging.getLogger()
        logger.addHandler(console_handler)
    
    def _check_js_runtime(self):
        """Deteksi JavaScript runtime"""
        for runtime in ['node', 'deno']:
            try:
                result = subprocess.run(
                    [runtime, '--version'],
                    capture_output=True,
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                if result.returncode == 0:
                    version = result.stdout.decode().strip()
                    self.safe_print(f"✓ JS runtime: {runtime.upper()} {version}")
                    return runtime
            except:
                continue
        
        self.safe_print("⚠️  JS runtime tidak ditemukan")
        return None
    
    def _load_config(self):
        """Load atau buat konfigurasi (untuk reload)"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    config = {**DEFAULT_CONFIG, **user_config}
            else:
                config = DEFAULT_CONFIG.copy()
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
            
            import shutil
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path and not config.get('ffmpeg_location'):
                config['ffmpeg_location'] = ffmpeg_path
            
            return config
            
        except Exception as e:
            self.safe_print(f"⚠️  Config error: {e}")
            return DEFAULT_CONFIG
            
    def save_config(self):
        """Save config ke file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            self.safe_print("✅ Konfigurasi disimpan")
        except Exception as e:
            self.safe_print(f"❌ Gagal simpan config: {e}")
    
    def validate_youtube_url(self, url: str) -> bool:
        """Validasi URL YouTube"""
        if not url or not isinstance(url, str):
            return False
        
        youtube_patterns = [
            r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+',
            r'(https?://)?(m\.)?youtube\.com/.+',
            r'(https?://)?youtu\.be/.+',
        ]
        
        for pattern in youtube_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def get_video_info(self, url: str, max_retries: int = 3):
        """Ekstrak informasi video dengan detail - FIXED VERSION"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'socket_timeout': self.config['timeout'],
            'extract_flat': False,
        }
        
        if self.js_runtime:
            ydl_opts['jsruntimes'] = self.js_runtime
        
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    if info:
                        # Safe method untuk mendapatkan format
                        formats = info.get('formats', [])
                        
                        # Helper function untuk mendapatkan nilai height dengan aman
                        def get_safe_height(fmt):
                            height = fmt.get('height')
                            if height is None:
                                return 0
                            try:
                                # Handle jika height adalah string seperti "720p"
                                if isinstance(height, str):
                                    # Extract numbers dari string
                                    numbers = re.findall(r'\d+', height)
                                    if numbers:
                                        return int(numbers[0])
                                    else:
                                        return 0
                                return int(height)
                            except (ValueError, TypeError):
                                return 0
                        
                        # Helper function untuk mendapatkan audio bitrate dengan aman
                        def get_safe_abr(fmt):
                            abr = fmt.get('abr')
                            if abr is None:
                                return 0.0
                            try:
                                if isinstance(abr, str):
                                    numbers = re.findall(r'\d+', abr)
                                    if numbers:
                                        return float(numbers[0])
                                    else:
                                        return 0.0
                                return float(abr)
                            except (ValueError, TypeError):
                                return 0.0
                        
                        # Filter format dengan aman
                        video_formats = [f for f in formats if f.get('vcodec') != 'none']
                        audio_formats = [f for f in formats if f.get('acodec') != 'none']
                        
                        # Cari best video format dengan error handling
                        best_video = None
                        if video_formats:
                            try:
                                best_video = max(video_formats, key=lambda x: get_safe_height(x))
                            except Exception:
                                # Jika gagal, gunakan yang pertama
                                best_video = video_formats[0] if video_formats else None
                        
                        # Cari best audio format dengan error handling
                        best_audio = None
                        if audio_formats:
                            try:
                                best_audio = max(audio_formats, key=lambda x: get_safe_abr(x))
                            except Exception:
                                best_audio = audio_formats[0] if audio_formats else None
                        
                        info['available_formats'] = {
                            'best_video': f"{get_safe_height(best_video)}p {best_video.get('ext', '') if best_video else 'N/A'}", 
                            'best_audio': f"{get_safe_abr(best_audio)}kbps {best_audio.get('ext', '') if best_audio else 'N/A'}",
                            'total_formats': len(formats),
                        }
                    
                    return info
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.safe_print(f"⏳ Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    self.safe_print(f"❌ Gagal get info: {e}")
                    return None
        return None
    
    def download_single(self, url: str, custom_format: str = None) -> bool:
        """Download single video dengan kualitas maksimal - SIMPLIFIED VERSION"""
        try:
            if self.config['validate_urls'] and not self.validate_youtube_url(url):
                self.safe_print(f"❌ URL tidak valid: {url[:50]}")
                return False
            
            self.safe_print(f"\n{'='*60}")
            self.safe_print(f"🔍 Mengambil info...")
            
            info = self.get_video_info(url, self.config['max_retries'])
            if not info:
                self.safe_print(f"❌ Gagal get info")
                return False
            
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            channel = info.get('channel', 'Unknown')
            view_count = info.get('view_count', 0)
            
            self.safe_print(f"📺 Judul: {title[:80]}")
            self.safe_print(f"👤 Channel: {channel}")
            self.safe_print(f"⏱️  Durasi: {self.video_processor._seconds_to_time(duration)}")
            if view_count:
                self.safe_print(f"👁️  Views: {view_count:,}")
            
            if 'available_formats' in info:
                fmt = info['available_formats']
                self.safe_print(f"📊 Format tersedia: {fmt.get('best_video', 'N/A')} + {fmt.get('best_audio', 'N/A')}")
            
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            safe_title = safe_title[:100]
            output_template = f'{safe_title}.%(ext)s'
            
            download_format = custom_format or self.config['default_format']
            
            # Simplified ydl_opts untuk menghindari error
            ydl_opts = {
                'format': download_format,
                'outtmpl': os.path.join(self.downloads_dir, output_template),
                'quiet': False,
                'no_warnings': True,
                'merge_output_format': 'mp4',
                'continuedl': self.config['resume_downloads'],
                'noprogress': False,
                'progress_hooks': [self._progress_hook],
            }
            
            # Only add features that are likely to work
            try:
                if self.config.get('embed_thumbnail'):
                    ydl_opts['writethumbnail'] = True
            except:
                pass
            
            self.safe_print(f"\n📥 Mendownload: {download_format}")
            self.safe_print(f"📁 Output: {self.downloads_dir}")
            
            start_time = datetime.now()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.download([url])
            
            end_time = datetime.now()
            download_time = (end_time - start_time).total_seconds()
            
            if result == 0:
                downloaded_files = self.video_processor.list_video_files(self.downloads_dir)
                if downloaded_files:
                    latest_file = max(downloaded_files, key=lambda x: Path(x).stat().st_mtime)
                    
                    file_size = Path(latest_file).stat().st_size / (1024 * 1024)
                    
                    self.safe_print(f"\n✅ Download selesai!")
                    self.safe_print(f"📁 File: {Path(latest_file).name}")
                    self.safe_print(f"📊 Size: {file_size:.1f}MB")
                    self.safe_print(f"⏱️  Waktu: {download_time:.1f}s")
                    
                    self._log_download(info, 'completed', file_size, download_time)
                    return True
                else:
                    self.safe_print("⚠️  File tidak ditemukan setelah download")
                    return False
            else:
                self.safe_print(f"❌ Download gagal")
                self._log_download(info, 'failed', 0, download_time)
                return False
        
        except Exception as e:
            self.safe_print(f"❌ Error: {str(e)[:200]}")
            if self.config.get('debug_mode'):
                traceback.print_exc()
            return False
    
    def _progress_hook(self, d):
        """Progress hook untuk yt-dlp"""
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            
            if total:
                percent = (downloaded / total) * 100
                speed = d.get('speed', 0)
                speed_mb = speed / (1024 * 1024) if speed else 0
                
                if hasattr(self, '_last_percent'):
                    if percent - self._last_percent >= 5:
                        self.safe_print(f"   Progress: {percent:.1f}% | Speed: {speed_mb:.1f} MB/s")
                        self._last_percent = percent
                else:
                    self._last_percent = 0
        
        elif d['status'] == 'finished':
            self.safe_print("   Processing video...")
    
    def batch_download(self, urls_file: Optional[str] = None) -> bool:
        """Download batch dengan manajemen yang lebih baik"""
        urls_file = urls_file or self.urls_file
        
        if not os.path.exists(urls_file):
            self.safe_print(f"❌ File tidak ditemukan: {urls_file}")
            return False
        
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        valid_urls = [url for url in urls if self.validate_youtube_url(url)]
        invalid_urls = [url for url in urls if not self.validate_youtube_url(url)]
        
        if invalid_urls:
            self.safe_print(f"⚠️  {len(invalid_urls)} URL tidak valid, dilewati")
            for url in invalid_urls[:3]:
                self.safe_print(f"   {url[:80]}")
        
        if not valid_urls:
            self.safe_print("❌ Tidak ada URL valid!")
            return False
        
        self.safe_print(f"\n📦 {len(valid_urls)} URL valid ditemukan")
        self.safe_print(f"⚙️  Max concurrent: {self.config['max_concurrent_downloads']}")
        
        successful = 0
        failed = 0
        
        max_workers = min(self.config['max_concurrent_downloads'], len(valid_urls), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.download_single, url): url for url in valid_urls}
            
            for i, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                self.safe_print(f"\n[{i}/{len(valid_urls)}] Processing: {url[:60]}...")
                
                try:
                    if future.result(timeout=3600):
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    self.safe_print(f"❌ Error: {e}")
                    failed += 1
        
        self.safe_print(f"\n{'='*60}")
        self.safe_print(f"📊 HASIL AKHIR:")
        self.safe_print(f"   ✅ Berhasil: {successful}")
        self.safe_print(f"   ❌ Gagal: {failed}")
        self.safe_print(f"   📁 Total: {len(valid_urls)}")
        
        return successful > 0
    
    # ========== VIDEO CLIPPING INTERFACE ==========
    def video_clipping_menu(self):
        """Menu video clipping yang lebih lengkap"""
        while True:
            print("\n" + "="*60)
            print("🎬 VIDEO CLIPPING MENU - ULTRA QUALITY")
            print("="*60)
            print("1. Potong video manual")
            print("2. Auto-detect & potong")
            print("3. Gabungkan klip")
            print("4. Ekstrak audio (High Quality)")
            print("5. List video tersedia dengan detail")
            print("6. Optimize video untuk web")
            print("7. Cek kualitas video")
            print("8. Settings kualitas")
            print("9. Kembali")
            
            choice = input("\nPilihan (1-9): ").strip()
            
            if choice == '1':
                self._manual_clip_video()
            elif choice == '2':
                self._auto_clip_video()
            elif choice == '3':
                self._merge_clips()
            elif choice == '4':
                self._extract_audio_advanced()
            elif choice == '5':
                self._list_videos_detailed()
            elif choice == '6':
                self._optimize_video()
            elif choice == '7':
                self._check_video_quality()
            elif choice == '8':
                self._quality_settings()
            elif choice == '9':
                break
            else:
                print("❌ Pilihan tidak valid!")
    
    def _list_videos_detailed(self):
        """List video dengan detail lengkap"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("📭 Tidak ada video")
            return
        
        print(f"\n📁 {len(videos)} video tersedia:")
        print("-" * 100)
        print(f"{'No.':<4} {'Filename':<40} {'Size':<10} {'Duration':<12} {'Resolution':<15}")
        print("-" * 100)
        
        for i, video in enumerate(videos[:20], 1):
            details = self.video_processor.get_video_details(video)
            if details:
                print(f"{i:<4} {details['filename'][:38]:<40} "
                      f"{details['size_mb']:<10.1f}MB "
                      f"{self.video_processor._seconds_to_time(details['duration']):<12} "
                      f"{details['resolution']:<15}")
        
        if len(videos) > 20:
            print(f"... dan {len(videos) - 20} video lainnya")
        print("-" * 100)
    
    def _manual_clip_video(self):
        """Manual video clipping dengan quality options"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("❌ Tidak ada video!")
            return
        
        print("\n📁 Pilih video:")
        for i, video in enumerate(videos[:10], 1):
            name = Path(video).name
            print(f"{i}. {name[:60]}")
        
        try:
            choice = int(input(f"\nPilih (1-{min(10, len(videos))}): ").strip())
            selected = videos[choice - 1]
        except:
            print("❌ Input tidak valid!")
            return
        
        details = self.video_processor.get_video_details(selected)
        if details:
            print(f"\n📺 Video details:")
            print(f"   Filename: {details['filename']}")
            print(f"   Size: {details['size_mb']:.1f}MB")
            print(f"   Duration: {self.video_processor._seconds_to_time(details['duration'])}")
            print(f"   Resolution: {details['resolution']}")
            print(f"   Video: {details['video_codec']}")
        
        print("\n🎚️  Pilih preset kualitas:")
        for name, profile in VideoProcessor.QUALITY_PROFILES.items():
            print(f"   {name:12} - {profile.description}")
        
        quality_preset = input("\nPreset (default: high_quality): ").strip() or "high_quality"
        
        segments = []
        print("\n✂️  Masukkan segment (ketik 'selesai' untuk stop):")
        
        while True:
            print(f"\nSegment {len(segments) + 1}:")
            start = input("  Waktu mulai (HH:MM:SS atau detik): ").strip()
            if start.lower() == 'selesai':
                break
            
            end = input("  Waktu akhir (HH:MM:SS atau detik): ").strip()
            if end.lower() == 'selesai':
                break
            
            title = input("  Judul (opsional): ").strip()
            
            segments.append(ClipSegment(
                start_time=start,
                end_time=end,
                title=title,
                quality_preset=quality_preset
            ))
        
        if segments:
            print(f"\n⚙️  Memproses {len(segments)} segment dengan preset '{quality_preset}'...")
            results = self.video_processor.clip_video(selected, segments)
            
            if results:
                total_size = sum(Path(f).stat().st_size for f in results) / (1024 * 1024)
                print(f"\n✅ Selesai! {len(results)} segment dibuat ({total_size:.1f}MB total)")
            else:
                print("❌ Gagal memproses segment")
    
    def _auto_clip_video(self):
        """Auto-clip video dengan options"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("❌ Tidak ada video!")
            return
        
        print("\n📁 Pilih video:")
        for i, video in enumerate(videos[:5], 1):
            print(f"{i}. {Path(video).name[:60]}")
        
        try:
            choice = int(input(f"\nPilih (1-{min(5, len(videos))}): ").strip())
            selected = videos[choice - 1]
        except:
            print("❌ Input tidak valid!")
            return
        
        print("\n🔍 Pilih metode deteksi:")
        print("   1. Equal segments (dibagi merata)")
        print("   2. Detect silence (segment berdasarkan silence)")
        
        method_choice = input("Pilihan (1-2): ").strip()
        method = "equal" if method_choice == "1" else "silence"
        
        min_dur = input("Durasi min (detik, default: 30): ").strip()
        max_dur = input("Durasi max (detik, default: 300): ").strip()
        
        min_dur = int(min_dur) if min_dur else 30
        max_dur = int(max_dur) if max_dur else 300
        
        if method == "equal":
            segments = self.video_processor.auto_detect_segments(selected, min_dur, max_dur, "equal")
        else:
            print("⚠️  Silence detection belum diimplementasikan, menggunakan equal segments")
            segments = self.video_processor.auto_detect_segments(selected, min_dur, max_dur, "equal")
        
        if segments:
            print(f"\n✅ Terdeteksi {len(segments)} segment")
            
            confirm = input("\nProses? (y/n): ").lower()
            if confirm == 'y':
                quality_preset = input("Preset kualitas (default: high_quality): ").strip() or "high_quality"
                for seg in segments:
                    seg.quality_preset = quality_preset
                
                results = self.video_processor.clip_video(selected, segments)
                print(f"\n✅ {len(results)} segment dibuat")
    
    def _merge_clips(self):
        """Merge klip dengan quality options"""
        clips = self.video_processor.list_video_files(str(self.video_processor.clips_dir))
        
        if len(clips) < 2:
            print("❌ Butuh minimal 2 klip di folder 'clips'!")
            return
        
        print("\n📁 Pilih klip (pisah dengan koma, atau 'all' untuk semua):")
        for i, clip in enumerate(clips[:15], 1):
            print(f"{i:2d}. {Path(clip).name[:60]}")
        
        try:
            choices_input = input("\nPilihan: ").strip()
            if choices_input.lower() == 'all':
                selected = clips
            else:
                choices = choices_input.split(',')
                selected = [clips[int(c.strip()) - 1] for c in choices]
        except:
            print("❌ Input tidak valid!")
            return
        
        print("\n🎚️  Pilih preset kualitas untuk merge:")
        for name in ['medium', 'high_quality', 'youtube']:
            profile = VideoProcessor.QUALITY_PROFILES[name]
            print(f"   {name:12} - {profile.description}")
        
        quality_preset = input("\nPreset (default: high_quality): ").strip() or "high_quality"
        
        output_name = input("\n📝 Nama output (tanpa ekstensi): ").strip() or "merged"
        output_path = str(self.video_processor.clips_dir / f"{output_name}_{quality_preset}.mp4")
        
        print(f"\n⚙️  Menggabungkan {len(selected)} klip dengan preset '{quality_preset}'...")
        
        original_preset = self.video_processor.quality_profile.name
        self.video_processor.quality_profile = self.video_processor.QUALITY_PROFILES[quality_preset]
        
        result = self.video_processor.merge_clips(selected, output_path)
        
        self.video_processor.quality_profile = self.video_processor.QUALITY_PROFILES[original_preset]
        
        if result:
            size_mb = Path(result).stat().st_size / (1024 * 1024)
            print(f"✅ Berhasil: {Path(result).name} ({size_mb:.1f}MB)")
    
    def _extract_audio_advanced(self):
        """Ekstrak audio dengan options lengkap"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("❌ Tidak ada video!")
            return
        
        print("\n📁 Pilih video:")
        for i, video in enumerate(videos[:5], 1):
            print(f"{i}. {Path(video).name[:60]}")
        
        try:
            choice = int(input(f"\nPilih (1-{min(5, len(videos))}): ").strip())
            selected = videos[choice - 1]
        except:
            print("❌ Input tidak valid!")
            return
        
        print(f"\n🎵 {Path(selected).name}")
        
        print("\n🎚️  Format audio:")
        print("   mp3  - Kompresi bagus, universal")
        print("   flac - Lossless, kualitas terbaik")
        print("   wav  - Uncompressed, kualitas maksimal")
        print("   m4a  - Apple format, kualitas bagus")
        
        format_choice = input("\nFormat (default: mp3): ").strip() or 'mp3'
        
        if format_choice == 'mp3':
            print("\n🎚️  Bitrate MP3:")
            print("   128k - Standar")
            print("   192k - Baik")
            print("   256k - Sangat baik")
            print("   320k - Terbaik")
            bitrate = input("Bitrate (default: 320k): ").strip() or '320k'
        elif format_choice == 'flac':
            bitrate = 'lossless'
        else:
            bitrate = input(f"Bitrate untuk {format_choice} (default: 256k): ").strip() or '256k'
        
        start = input("\nWaktu mulai (HH:MM:SS atau detik): ").strip()
        end = input("Waktu akhir (HH:MM:SS atau detik): ").strip()
        
        config = AudioExtractConfig(
            start_time=start,
            end_time=end,
            output_format=format_choice,
            bitrate=bitrate,
            sample_rate='48000',
            channels=2,
            normalize_audio=True,
            remove_noise=False
        )
        
        print(f"\n⚙️  Mengekstrak audio {format_choice.upper()} @ {bitrate}...")
        result = self.video_processor.extract_audio_segment(selected, config)
        
        if result:
            print(f"✅ Audio berhasil diekstrak")
    
    def _optimize_video(self):
        """Optimize video untuk web/streaming"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("❌ Tidak ada video!")
            return
        
        print("\n📁 Pilih video untuk dioptimalkan:")
        for i, video in enumerate(videos[:5], 1):
            size_mb = Path(video).stat().st_size / (1024 * 1024)
            print(f"{i}. {Path(video).name[:50]} ({size_mb:.1f}MB)")
        
        try:
            choice = int(input(f"\nPilih (1-{min(5, len(videos))}): ").strip())
            selected = videos[choice - 1]
        except:
            print("❌ Input tidak valid!")
            return
        
        output_path = str(Path(selected).parent / f"optimized_{Path(selected).name}")
        
        print(f"\n⚙️  Mengoptimalkan video...")
        
        try:
            ffmpeg_cmd = ['ffmpeg']
            ffmpeg_cmd.extend([
                '-i', selected,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                '-y',
                output_path
            ])
            
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            
            if Path(output_path).exists():
                orig_size = Path(selected).stat().st_size / (1024 * 1024)
                new_size = Path(output_path).stat().st_size / (1024 * 1024)
                reduction = ((orig_size - new_size) / orig_size) * 100
                
                print(f"✅ Optimasi selesai!")
                print(f"   Original: {orig_size:.1f}MB")
                print(f"   Optimized: {new_size:.1f}MB")
                print(f"   Reduction: {reduction:.1f}%")
        except Exception as e:
            print(f"❌ Gagal optimize: {e}")
    
    def _check_video_quality(self):
        """Cek kualitas video secara detail"""
        videos = self.video_processor.list_video_files(self.downloads_dir)
        
        if not videos:
            print("❌ Tidak ada video!")
            return
        
        print("\n📁 Pilih video:")
        for i, video in enumerate(videos[:5], 1):
            print(f"{i}. {Path(video).name[:60]}")
        
        try:
            choice = int(input(f"\nPilih (1-{min(5, len(videos))}): ").strip())
            selected = videos[choice - 1]
        except:
            print("❌ Input tidak valid!")
            return
        
        info = self.video_processor.get_video_info(selected)
        if not info:
            print("❌ Tidak bisa membaca video info")
            return
        
        print(f"\n📊 DETAIL KUALITAS VIDEO:")
        print(f"   File: {Path(selected).name}")
        size_mb = int(info.get('size', 0)) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f}MB")
        print(f"   Duration: {self.video_processor._seconds_to_time(info.get('duration', 0))}")
        print(f"   Bitrate: {int(info.get('bitrate', 0))//1000 if info.get('bitrate') else '?'} Kbps")
        print(f"   Format: {info.get('format_name', 'Unknown')}")
        
        print(f"\n🎬 VIDEO STREAM:")
        print(f"   Codec: {info.get('video_codec', 'Unknown')}")
        print(f"   Resolution: {info.get('video_width', 0)}x{info.get('video_height', 0)}")
        print(f"   Bitrate: {int(info.get('video_bitrate', 0))//1000 if info.get('video_bitrate') else '?'} Kbps")
        print(f"   FPS: {info.get('video_fps', 0):.2f}")
        
        print(f"\n🔊 AUDIO STREAM:")
        print(f"   Codec: {info.get('audio_codec', 'Unknown')}")
        print(f"   Bitrate: {int(info.get('audio_bitrate', 0))//1000 if info.get('audio_bitrate') else '?'} Kbps")
        print(f"   Sample Rate: {info.get('audio_sample_rate', 'Unknown')} Hz")
    
    def _quality_settings(self):
        """Ubah pengaturan kualitas"""
        while True:
            print("\n" + "="*60)
            print("⚙️  SETTINGS KUALITAS")
            print("="*60)
            print("1. Ubah preset clipping")
            print("2. Ubah format download")
            print("3. Ubah audio settings")
            print("4. Reset ke default")
            print("5. Tampilkan current settings")
            print("6. Kembali")
            
            choice = input("\nPilihan (1-6): ").strip()
            
            if choice == '1':
                print("\n🎚️  Preset clipping saat ini:", self.config.get('clipping_preset'))
                print("Available presets:")
                for name, profile in VideoProcessor.QUALITY_PROFILES.items():
                    print(f"   {name:12} - CRF: {profile.crf:3} | Preset: {profile.preset:10} | {profile.description}")
                
                new_preset = input("\nPreset baru: ").strip()
                if new_preset in VideoProcessor.QUALITY_PROFILES:
                    self.config['clipping_preset'] = new_preset
                    self.video_processor.quality_profile = VideoProcessor.QUALITY_PROFILES[new_preset]
                    print(f"✅ Preset diubah ke: {new_preset}")
                else:
                    print("❌ Preset tidak valid!")
            
            elif choice == '2':
                print("\n📥 Format download saat ini:", self.config.get('default_format'))
                print("\nContoh format:")
                print("   best[height<=1080]  # 1080p")
                print("   best[height<=720]   # 720p")
                print("   best[ext=mp4]       # MP4 terbaik")
                print("   bestaudio           # Audio saja")
                
                new_format = input("\nFormat baru: ").strip()
                if new_format:
                    self.config['default_format'] = new_format
                    print(f"✅ Format diubah ke: {new_format}")
            
            elif choice == '3':
                print("\n🔊 Audio settings saat ini:")
                print(f"   Bitrate: {self.config.get('audio_bitrate')}")
                print(f"   Sample rate: {self.config.get('audio_sample_rate')}")
                print(f"   Codec: {self.config.get('audio_codec')}")
                
                new_bitrate = input("\nAudio bitrate baru (contoh: 320k): ").strip()
                if new_bitrate:
                    self.config['audio_bitrate'] = new_bitrate
                
                new_sr = input("Sample rate baru (contoh: 48000): ").strip()
                if new_sr:
                    self.config['audio_sample_rate'] = new_sr
                
                print("✅ Audio settings diupdate")
            
            elif choice == '4':
                confirm = input("\nReset semua settings ke default? (y/n): ").lower()
                if confirm == 'y':
                    self.config = DEFAULT_CONFIG.copy()
                    self.config['download_path'] = self.downloads_dir
                    print("✅ Settings direset ke default")
            
            elif choice == '5':
                print("\n📋 CURRENT SETTINGS:")
                for key, value in self.config.items():
                    if not isinstance(value, (dict, list)) or (isinstance(value, list) and len(value) <= 5):
                        print(f"   {key:30}: {value}")
                
                print(f"\n🎬 QUALITY PROFILE: {self.video_processor.quality_profile.name}")
                profile = self.video_processor.quality_profile
                print(f"   CRF: {profile.crf} | Preset: {profile.preset}")
                print(f"   Tune: {profile.tune} | Profile: {profile.profile}")
                print(f"   Description: {profile.description}")
            
            elif choice == '6':
                self.save_config()
                break
    
    def _log_download(self, info: dict, status: str, file_size: float = 0, duration: float = 0):
        """Log download history dengan detail"""
        try:
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'title': info.get('title', 'Unknown')[:100],
                'channel': info.get('channel', 'Unknown'),
                'url': info.get('webpage_url', ''),
                'duration': info.get('duration', 0),
                'status': status,
                'file_size_mb': round(file_size, 1),
                'download_time_seconds': round(duration, 1),
                'format': self.config.get('default_format', ''),
            }
            
            history.append(log_entry)
            
            if len(history) > 100:
                history = history[-100:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Gagal log history: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.current_downloads.clear()
        self.video_processor.cleanup_temp_files()
        self.printer.stop()


# ==================== UI FUNCTIONS ====================
def show_history(history_file: str):
    """Tampilkan download history dengan detail"""
    try:
        if not os.path.exists(history_file):
            print("📭 Belum ada history")
            return
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            print("📭 History kosong")
            return
        
        print(f"\n📜 DOWNLOAD HISTORY ({len(history)} entri):")
        print("=" * 100)
        print(f"{'No.':<4} {'Tanggal':<16} {'Status':<8} {'Title':<40} {'Size':<8} {'Time':<6}")
        print("=" * 100)
        
        for i, entry in enumerate(reversed(history[-20:]), 1):
            status = '✅' if entry['status'] == 'completed' else '❌'
            title = entry['title'][:38] + '..' if len(entry['title']) > 40 else entry['title']
            date = entry['timestamp'][:16].replace('T', ' ')
            
            print(f"{i:<4} {date:<16} {status:<8} {title:<40} "
                  f"{entry.get('file_size_mb', 0):<8.1f}MB "
                  f"{entry.get('download_time_seconds', 0):<6.1f}s")
        
        print("=" * 100)
        
        completed = sum(1 for e in history if e['status'] == 'completed')
        failed = sum(1 for e in history if e['status'] == 'failed')
        total_size = sum(e.get('file_size_mb', 0) for e in history if e['status'] == 'completed')
        
        print(f"📊 Statistik: ✅ {completed} | ❌ {failed} | 📁 {total_size:.1f}MB total")
        
    except Exception as e:
        print(f"❌ Gagal baca history: {e}")


def interactive_mode(downloader: YouTubeDownloader):
    """Mode interaktif yang lebih lengkap"""
    while True:
        print("\n" + "="*60)
        print("🚀 YOUTUBE DOWNLOADER PRO - ULTRA QUALITY")
        print("="*60)
        print("1. Download single URL")
        print("2. Batch download dari file")
        print("3. Video Clipping (Advanced)")
        print("4. History & Statistics")
        print("5. Tools & Utilities")
        print("6. Settings")
        print("7. Cleanup temporary files")
        print("8. Test FFmpeg")
        print("9. Keluar")
        
        choice = input("\nPilihan (1-9): ").strip()
        
        if choice == '1':
            url = input("URL YouTube: ").strip()
            if url:
                print("\n📥 Format download (kosong untuk default):")
                print("Contoh: best[height<=1080]")
                custom_format = input("Format custom: ").strip()
                downloader.download_single(url, custom_format if custom_format else None)
        
        elif choice == '2':
            urls_file = input("Path ke file URLs (kosong untuk default): ").strip()
            if not urls_file:
                urls_file = downloader.urls_file
            
            if os.path.exists(urls_file):
                downloader.batch_download(urls_file)
            else:
                print(f"❌ File tidak ditemukan: {urls_file}")
                create_new = input("Buat file baru? (y/n): ").lower()
                if create_new == 'y':
                    with open(urls_file, 'w', encoding='utf-8') as f:
                        f.write("# Masukkan URL YouTube (satu per baris)\n")
                        f.write("# https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
                    print(f"✅ File dibuat: {urls_file}")
        
        elif choice == '3':
            downloader.video_clipping_menu()
        
        elif choice == '4':
            show_history(downloader.history_file)
        
        elif choice == '5':
            print("\n🛠️  TOOLS & UTILITIES")
            print("1. List semua video dengan detail")
            print("2. Cek kualitas video")
            print("3. Optimize video untuk web")
            
            tool_choice = input("Pilihan (1-3): ").strip()
            
            if tool_choice == '1':
                downloader._list_videos_detailed()
            elif tool_choice == '2':
                downloader._check_video_quality()
            elif tool_choice == '3':
                downloader._optimize_video()
        
        elif choice == '6':
            downloader._quality_settings()
        
        elif choice == '7':
            confirm = input("Bersihkan semua file temporary? (y/n): ").lower()
            if confirm == 'y':
                downloader.video_processor.cleanup_temp_files()
                print("✅ Temporary files dibersihkan")
        
        elif choice == '8':
            print("\n🔧 TESTING FFMPEG...")
            if downloader.video_processor.ffmpeg_available:
                print("✅ FFmpeg berfungsi dengan baik")
            else:
                print("❌ FFmpeg tidak tersedia")
        
        elif choice == '9':
            print("\n👋 Keluar dari program...")
            break
        
        else:
            print("❌ Pilihan tidak valid!")


# ==================== MAIN ====================
def check_dependencies():
    """Cek dependencies yang diperlukan secara detail"""
    missing = []
    warnings_list = []
    
    print("🔧 CHECKING DEPENDENCIES...")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version_line}")
        else:
            missing.append('ffmpeg')
    except:
        missing.append('ffmpeg')
    
    try:
        import yt_dlp
        print(f"✅ yt-dlp: {yt_dlp.version.__version__}")
    except:
        missing.append('yt-dlp')
    
    if missing:
        print("\n❌ DEPENDENCIES YANG BELUM TERINSTALL:")
        for dep in missing:
            print(f"  - {dep}")
        
        print("\n📦 INSTALLATION GUIDE:")
        if 'ffmpeg' in missing:
            print("  FFmpeg: https://ffmpeg.org/download.html")
            print("  Windows: winget install ffmpeg")
        
        if 'yt-dlp' in missing:
            print("  yt-dlp: pip install yt-dlp --upgrade")
        
        return False
    
    if warnings_list:
        print("\n⚠️  WARNINGS:")
        for warning in warnings_list:
            print(f"  - {warning}")
        print("\n⚠️  Program mungkin masih berjalan, tapi beberapa fitur mungkin tidak optimal.")
    
    print("\n✅ Semua dependencies terpenuhi!")
    return True

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 YOUTUBE DOWNLOADER PRO + VIDEO CLIPPER")
    print("   VERSION 2.0 - ULTRA QUALITY EDITION")
    print("="*60)
    print(f"📁 Base Directory: {BASE_PATH}")
    print("="*60)
    
    if not check_dependencies():
        input("\nInstall dependencies terlebih dahulu. Tekan Enter untuk keluar...")
        return
    
    try:
        downloader = YouTubeDownloader()
        
        if os.path.exists(downloader.urls_file):
            print(f"\n📁 URLs file ditemukan: {downloader.urls_file}")
            
            with open(downloader.urls_file, 'r', encoding='utf-8') as f:
                url_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
            
            if url_count > 0:
                print(f"📊 {url_count} URLs ditemukan dalam file")
                print("\nPilihan:")
                print("  1. Jalankan batch download")
                print("  2. Edit file URLs")
                print("  3. Masuk ke menu interaktif")
                
                choice = input("\nPilihan (1-3): ").strip()
                
                if choice == '1':
                    downloader.batch_download()
                elif choice == '2':
                    try:
                        os.system(f'notepad "{downloader.urls_file}"')
                    except:
                        print("⚠️  Tidak bisa membuka editor")
                    interactive_mode(downloader)
                else:
                    interactive_mode(downloader)
            else:
                print("⚠️  File URLs kosong")
                interactive_mode(downloader)
        else:
            print(f"\n⚠️  File tidak ditemukan: {downloader.urls_file}")
            
            with open(downloader.urls_file, 'w', encoding='utf-8') as f:
                f.write("# YOUTUBE DOWNLOADER PRO - URL LIST\n")
                f.write("# Masukkan URL YouTube (satu per baris)\n")
                f.write("# Contoh:\n")
                f.write("# https://www.youtube.com/watch?v=dQw4w9WgXcQ\n")
                f.write("# https://youtu.be/VIDEO_ID\n")
                f.write("\n")
            
            print(f"✅ File template dibuat: {downloader.urls_file}")
            print("\n1. Edit file URLs terlebih dahulu")
            print("2. Langsung ke menu interaktif")
            
            choice = input("\nPilihan (1-2): ").strip()
            
            if choice == '1':
                try:
                    os.system(f'notepad "{downloader.urls_file}"')
                except:
                    print("⚠️  Tidak bisa membuka editor")
                interactive_mode(downloader)
            else:
                interactive_mode(downloader)
    
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan oleh user")
    
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
    
    finally:
        try:
            if 'downloader' in locals():
                downloader.cleanup()
        except:
            pass
        
        print("\n✨ Program selesai. Goodbye!")
        input("Tekan Enter untuk menutup...")

if __name__ == "__main__":
    main()