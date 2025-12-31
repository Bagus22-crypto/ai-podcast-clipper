"""
YouTube Downloader Pro - Enhanced & Cleaned Version
Author: Enhanced by AI
Description: Multi-threaded YouTube video downloader with resume capability
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
import socket
import urllib.parse
import traceback

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
BASE_PATH = r"C:\Users\BAGUS\ai-podcast-clipper\youtube_download"

DEFAULT_CONFIG = {
    'output_template': '%(title)s.%(ext)s',
    'max_concurrent_downloads': 2,
    'default_format': 'best[height<=720]',
    'download_subtitles': False,
    'subtitles_langs': ['en'],
    'embed_metadata': False,
    'embed_thumbnail': False,
    'auto_convert_audio': 'mp3',
    'limit_rate': None,
    'retry_count': 3,
    'max_retries': 5,
    'ignore_errors': True,
    'resume_downloads': True,
    'timeout': 30,
    'validate_urls': True,
    'suppress_warnings': True,
    'use_android_client': False,
    'move_to_completed': False,
    'debug_mode': False,
}


# ==================== THREAD-SAFE PRINTER ====================
class ThreadSafePrinter:
    """Thread-safe printer untuk output yang bersih dan teratur"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = Queue()
        self._stop_event = threading.Event()
        self._printer_thread = threading.Thread(target=self._print_worker, daemon=True)
        self._printer_thread.start()
    
    def _print_worker(self):
        """Worker thread untuk mencetak output"""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=1)
                if item:
                    with self._lock:
                        print(*item if isinstance(item, tuple) else (item,), flush=True)
                    self._queue.task_done()
            except:
                pass
    
    def print(self, *args):
        """Thread-safe print"""
        self._queue.put(args)
    
    def stop(self):
        """Stop printer thread"""
        self._stop_event.set()
        self._printer_thread.join(timeout=2)


# ==================== YOUTUBE DOWNLOADER ====================
class YouTubeDownloader:
    """Main downloader class dengan semua fungsi download"""
    
    def __init__(self):
        self.base_path = BASE_PATH
        self._setup_paths()
        self.config = self._load_config()
        self.printer = ThreadSafePrinter()
        self.current_downloads = {}
        self.running = True
        self.js_runtime = self._check_js_runtime()
    
    # ========== SETUP & INITIALIZATION ==========
    def _setup_paths(self):
        """Setup semua path yang diperlukan"""
        os.makedirs(self.base_path, exist_ok=True)
        
        self.history_file = os.path.join(self.base_path, 'download_history.json')
        self.config_file = os.path.join(self.base_path, 'config.json')
        self.status_file = os.path.join(self.base_path, 'download_status.json')
        self.downloads_dir = os.path.join(self.base_path, 'downloads')
        self.urls_file = os.path.join(self.base_path, 'urls.txt')
        self.completed_dir = os.path.join(self.base_path, 'completed')
        self.temp_dir = os.path.join(self.base_path, 'temp')
        
        # Buat semua direktori
        for directory in [self.downloads_dir, self.completed_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _check_js_runtime(self):
        """Deteksi JavaScript runtime yang tersedia"""
        runtimes = ['node', 'deno', 'quickjs']
        
        for runtime in runtimes:
            try:
                result = subprocess.run(
                    [runtime, '--version'],
                    capture_output=True,
                    timeout=2,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                if result.returncode == 0:
                    self.safe_print(f"✓ JavaScript runtime: {runtime.upper()}")
                    return runtime
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        self.safe_print("⚠️  JavaScript runtime tidak ditemukan")
        self.safe_print("💡 Instal Node.js: https://nodejs.org/")
        return None
    
    def _load_config(self):
        """Load atau buat konfigurasi default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # Merge dengan default
                    return {**DEFAULT_CONFIG, **user_config, 'base_path': BASE_PATH}
            
            # Buat config baru
            config = {**DEFAULT_CONFIG, 'base_path': BASE_PATH, 'download_path': self.downloads_dir}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return config
            
        except Exception as e:
            self.safe_print(f"⚠️  Gagal load config: {e}")
            return DEFAULT_CONFIG
    
    def save_config(self):
        """Simpan konfigurasi"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.safe_print(f"❌ Gagal simpan config: {e}")
            return False
    
    # ========== URL VALIDATION ==========
    def validate_youtube_url(self, url):
        """Validasi URL YouTube"""
        if not url or not isinstance(url, str):
            return False
        
        # Cek domain YouTube
        try:
            parsed = urllib.parse.urlparse(url)
            valid_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
            return any(domain in parsed.netloc for domain in valid_domains)
        except:
            return False
    
    # ========== STATUS MANAGEMENT ==========
    def save_download_status(self, url, status_data):
        """Simpan status download"""
        try:
            all_status = {}
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    all_status = json.load(f)
            
            all_status[url] = {**status_data, 'timestamp': datetime.now().isoformat()}
            
            # Simpan hanya 100 status terakhir
            if len(all_status) > 100:
                sorted_items = sorted(all_status.items(), key=lambda x: x[1].get('timestamp', ''))
                all_status = dict(sorted_items[-100:])
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(all_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.safe_print(f"⚠️  Gagal simpan status: {e}")
    
    def get_download_status(self, url):
        """Ambil status download"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get(url)
        except:
            pass
        return None
    
    def clear_download_status(self, url):
        """Hapus status download"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    all_status = json.load(f)
                
                all_status.pop(url, None)
                
                with open(self.status_file, 'w', encoding='utf-8') as f:
                    json.dump(all_status, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    # ========== ERROR HANDLING ==========
    def handle_error(self, error):
        """Handle dan kategorikan error"""
        error_str = str(error).lower()
        
        error_map = {
            'quota exceeded': ('QUOTA_EXCEEDED', 'Kuota API terlampaui', 3600, True),
            'network': ('NETWORK_ERROR', 'Koneksi bermasalah', 30, True),
            'timeout': ('NETWORK_ERROR', 'Koneksi timeout', 30, True),
            'private video': ('PRIVATE_VIDEO', 'Video pribadi', None, False),
            'video unavailable': ('VIDEO_UNAVAILABLE', 'Video tidak tersedia', None, False),
            'age restricted': ('AGE_RESTRICTED', 'Video dibatasi usia', None, False),
            'sign in': ('SIGN_IN_REQUIRED', 'Perlu login', None, False),
            'format not available': ('FORMAT_UNAVAILABLE', 'Format tidak tersedia', None, True),
            '429': ('RATE_LIMIT', 'Terlalu banyak request', 300, True),
            'javascript': ('JS_RUNTIME_MISSING', 'JS runtime tidak ada', None, False),
        }
        
        for key, (error_type, message, retry_after, can_retry) in error_map.items():
            if key in error_str:
                return {
                    'type': error_type,
                    'message': message,
                    'retry_after': retry_after,
                    'can_retry': can_retry
                }
        
        return {
            'type': 'UNKNOWN_ERROR',
            'message': f'Error: {str(error)[:200]}',
            'retry_after': 60,
            'can_retry': True
        }
    
    # ========== FORMAT SELECTION ==========
    def get_video_info(self, url, max_retries=3):
        """Ekstrak informasi video dengan retry"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'socket_timeout': self.config['timeout'],
            'extractor_retries': 2,
        }
        
        if self.js_runtime:
            ydl_opts['jsruntimes'] = self.js_runtime
        
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=False)
            except Exception as e:
                if attempt < max_retries - 1:
                    self.safe_print(f"⏳ Retry {attempt + 1}/{max_retries}...")
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise
        return None
    
    def select_best_format(self, formats):
        """Pilih format terbaik (720p preferred)"""
        if not formats:
            return 'best[height<=720]'
        
        # Cari 720p
        for f in formats:
            if '720' in str(f.get('resolution', '')):
                return f.get('id', 'best[height<=720]')
        
        # Fallback ke format dengan video
        for f in formats:
            if f.get('has_video'):
                return f.get('id', 'best')
        
        return formats[0].get('id', 'best') if formats else 'best'
    
    def get_formats_info(self, info):
        """Ekstrak informasi format dari video info"""
        formats = []
        
        for f in info.get('formats', []):
            if f.get('vcodec') != 'none' or f.get('acodec') != 'none':
                filesize = f.get('filesize') or f.get('filesize_approx', 0)
                formats.append({
                    'id': f.get('format_id', 'N/A'),
                    'resolution': f.get('resolution', 'unknown'),
                    'ext': f.get('ext', 'mp4'),
                    'filesize': filesize,
                    'size_mb': filesize / (1024*1024) if filesize else 0,
                    'fps': f.get('fps', 0),
                    'vcodec': f.get('vcodec', 'unknown'),
                    'acodec': f.get('acodec', 'unknown'),
                    'note': f.get('format_note', ''),
                    'has_video': f.get('vcodec') != 'none',
                    'has_audio': f.get('acodec') != 'none',
                })
        
        return formats
    
    # ========== DOWNLOAD OPTIONS ==========
    def get_ydl_options(self, format_id, output_template):
        """Generate yt-dlp options"""
        abs_path = os.path.abspath(self.downloads_dir)
        
        options = {
            'format': format_id,
            'outtmpl': os.path.join(abs_path, output_template),
            'quiet': True,
            'no_warnings': self.config['suppress_warnings'],
            'continuedl': self.config['resume_downloads'],
            'retries': self.config['retry_count'],
            'fragment_retries': self.config['retry_count'],
            'skip_unavailable_fragments': True,
            'ignoreerrors': self.config['ignore_errors'],
            'merge_output_format': 'mp4',
            'socket_timeout': self.config['timeout'],
            'extractor_retries': 3,
            'progress_hooks': [self._create_progress_hook(format_id)],
            'paths': {
                'home': abs_path,
                'temp': self.temp_dir
            },
        }
        
        if self.js_runtime:
            options['jsruntimes'] = self.js_runtime
        
        if self.config.get('use_android_client'):
            options['extractor_args'] = {
                'youtube': {
                    'player_client': ['android'],
                    'player_skip': ['webpage'],
                }
            }
        
        # Subtitle options
        if self.config.get('download_subtitles'):
            options.update({
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': self.config['subtitles_langs'],
            })
        
        # Metadata options
        if self.config.get('embed_metadata'):
            options['addmetadata'] = True
        
        if self.config.get('embed_thumbnail'):
            options.update({
                'writethumbnail': True,
                'embedthumbnail': True,
            })
        
        return options
    
    def _create_progress_hook(self, format_id):
        """Buat progress hook untuk tracking"""
        def progress_hook(d):
            if not self.running:
                raise KeyboardInterrupt("Download dihentikan")
            
            if d['status'] == 'downloading':
                downloaded = d.get('downloaded_bytes', 0)
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                
                if total > 0:
                    percent = int((downloaded / total) * 100)
                    if percent % 10 == 0 and percent > 0:
                        if format_id not in self.current_downloads:
                            self.current_downloads[format_id] = set()
                        if percent not in self.current_downloads[format_id]:
                            self.safe_print(f"  📥 Progress: {percent}%")
                            self.current_downloads[format_id].add(percent)
            
            elif d['status'] == 'finished':
                self.safe_print(f"  ✓ Download selesai, memproses...")
        
        return progress_hook
    
    # ========== MAIN DOWNLOAD FUNCTION ==========
    def download_single(self, url):
        """Download single video"""
        thread_id = threading.get_ident()
        
        try:
            # Validasi URL
            if self.config['validate_urls'] and not self.validate_youtube_url(url):
                self.safe_print(f"❌ URL tidak valid: {url[:50]}")
                return False
            
            self.safe_print(f"\n{'='*60}")
            self.safe_print(f"🔍 Thread {thread_id}: Mengambil info...")
            
            # Get video info
            info = self.get_video_info(url, self.config['max_retries'])
            if not info:
                self.safe_print(f"❌ Gagal mendapatkan info dari {url}")
                return False
            
            # Tampilkan info dasar
            title = info.get('title', 'Unknown')[:50]
            uploader = info.get('uploader', 'Unknown')
            duration = info.get('duration', 0)
            
            self.safe_print(f"📺 {title}")
            self.safe_print(f"👤 {uploader}")
            if duration:
                self.safe_print(f"⏱️  {self._format_duration(duration)}")
            
            # Pilih format
            formats = self.get_formats_info(info)
            format_id = self.select_best_format(formats)
            
            # Setup output
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            output_template = f'{safe_title}.%(ext)s'
            
            # Prepare download
            options = self.get_ydl_options(format_id, output_template)
            
            self.safe_print(f"📥 Mendownload...")
            self.safe_print(f"🎯 Format: {format_id}")
            
            # Save status
            self.save_download_status(url, {
                'url': url,
                'title': title,
                'format': format_id,
                'started_at': datetime.now().isoformat(),
                'status': 'downloading',
            })
            
            # Download dengan retry
            success = False
            for attempt in range(self.config['max_retries']):
                try:
                    with yt_dlp.YoutubeDL(options) as ydl:
                        result = ydl.download([url])
                    
                    if result == 0:
                        success = True
                        break
                    
                except KeyboardInterrupt:
                    self.safe_print(f"⏸️  Download dihentikan")
                    self.save_download_status(url, {'status': 'paused'})
                    return False
                
                except Exception as e:
                    error_info = self.handle_error(e)
                    
                    if attempt < self.config['max_retries'] - 1 and error_info['can_retry']:
                        wait = error_info.get('retry_after', 5 * (attempt + 1))
                        self.safe_print(f"🔄 {error_info['message']}, retry dalam {wait}s...")
                        import time
                        time.sleep(wait)
                    else:
                        self.safe_print(f"❌ {error_info['message']}")
                        break
            
            # Handle result
            if success:
                self.safe_print(f"✅ Selesai: {title[:40]}")
                self._log_download(info, format_id, 'completed')
                self.clear_download_status(url)
                self._cleanup_partial_files(safe_title)
                return True
            else:
                self.safe_print(f"❌ Gagal: {title[:40]}")
                self._log_download(info, format_id, 'failed')
                return False
        
        except Exception as e:
            error_info = self.handle_error(e)
            self.safe_print(f"❌ {error_info['message']}")
            return False
    
    # ========== BATCH DOWNLOAD ==========
    def batch_download(self, urls_file=None):
        """Download batch dari file"""
        urls_file = urls_file or self.urls_file
        
        if not os.path.exists(urls_file):
            self.safe_print(f"❌ File tidak ditemukan: {urls_file}")
            return False
        
        # Load URLs
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Validasi
        valid_urls = [url for url in urls if self.validate_youtube_url(url)]
        
        if not valid_urls:
            self.safe_print("❌ Tidak ada URL valid!")
            return False
        
        self.safe_print(f"\n📦 {len(valid_urls)} URL valid ditemukan")
        self.safe_print(f"🚀 Menggunakan {self.config['max_concurrent_downloads']} thread")
        self.safe_print("-" * 60)
        
        # Reset
        self.current_downloads = {}
        self.running = True
        
        successful = 0
        failed = 0
        
        # Download dengan ThreadPoolExecutor
        max_workers = min(self.config['max_concurrent_downloads'], len(valid_urls), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.download_single, url): url for url in valid_urls}
            
            try:
                for future in as_completed(future_to_url, timeout=3600):
                    if not self.running:
                        break
                    
                    try:
                        if future.result(timeout=600):
                            successful += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
            
            except KeyboardInterrupt:
                self.safe_print("\n⚠️  Batch dihentikan!")
                self.running = False
                executor.shutdown(wait=False, cancel_futures=True)
        
        # Summary
        self.safe_print(f"\n{'='*60}")
        self.safe_print(f"🎉 SELESAI!")
        self.safe_print(f"✅ Berhasil: {successful}")
        self.safe_print(f"❌ Gagal: {failed}")
        self.safe_print(f"📊 Total: {len(valid_urls)}")
        
        self._cleanup_resources()
        return successful > 0
    
    # ========== HELPER FUNCTIONS ==========
    def _format_duration(self, seconds):
        """Format durasi"""
        if not seconds:
            return "Live"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes}:{seconds:02d}"
    
    def _log_download(self, info, format_choice, status):
        """Log download history"""
        try:
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append({
                'timestamp': datetime.now().isoformat(),
                'title': info.get('title', 'Unknown')[:100],
                'url': info.get('webpage_url', ''),
                'format': format_choice,
                'status': status,
            })
            
            # Keep last 50
            history = history[-50:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def _cleanup_partial_files(self, title):
        """Cleanup file .part"""
        try:
            for file in Path(self.downloads_dir).glob(f"{title}*.part"):
                file.unlink()
        except:
            pass
    
    def _cleanup_resources(self):
        """Cleanup resources"""
        self.running = False
        self.current_downloads.clear()
        import gc
        gc.collect()
    
    def safe_print(self, *args):
        """Thread-safe print"""
        self.printer.print(*args)
    
    def show_directory_structure(self):
        """Tampilkan struktur direktori"""
        self.safe_print(f"\n📁 STRUKTUR DIREKTORI: {self.base_path}")
        self.safe_print("=" * 60)
        
        for root, dirs, files in os.walk(self.base_path):
            level = root.replace(self.base_path, '').count(os.sep)
            indent = '  ' * level
            self.safe_print(f'{indent}📁 {os.path.basename(root)}/')
            
            subindent = '  ' * (level + 1)
            for file in files[:5]:
                self.safe_print(f'{subindent}📄 {file}')
            
            if len(files) > 5:
                self.safe_print(f'{subindent}... +{len(files) - 5} file')


# ==================== UI FUNCTIONS ====================
def show_history(history_file):
    """Tampilkan history download"""
    try:
        if not os.path.exists(history_file):
            print("📭 Belum ada history")
            return
        
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        print(f"\n📜 HISTORY ({len(history)} entri):")
        print("=" * 80)
        
        for i, entry in enumerate(reversed(history[-10:]), 1):
            status_icon = '✅' if entry['status'] == 'completed' else '❌'
            print(f"{i}. {entry['timestamp'][:16]} {status_icon} {entry['title'][:50]}")
        
        print("=" * 80)
    except:
        print("❌ Gagal baca history")


def interactive_mode(downloader):
    """Mode interaktif"""
    while True:
        print("\n🎯 MENU:")
        print("1. Download single URL")
        print("2. Batch download")
        print("3. Tampilkan history")
        print("4. Tampilkan struktur folder")
        print("5. Keluar")
        
        choice = input("\nPilihan (1-5): ").strip()
        
        if choice == '1':
            url = input("URL YouTube: ").strip()
            if url:
                downloader.download_single(url)
        
        elif choice == '2':
            downloader.batch_download()
        
        elif choice == '3':
            show_history(downloader.history_file)
        
        elif choice == '4':
            downloader.show_directory_structure()
        
        elif choice == '5':
            break
        
        else:
            print("❌ Pilihan tidak valid!")


# ==================== MAIN ====================
def main():
    """Main function"""
    print("🚀 YouTube Downloader Pro")
    print(f"📁 Base: {BASE_PATH}")
    print("=" * 60)
    
    downloader = YouTubeDownloader()
    
    try:
        # Cek file urls.txt
        if os.path.exists(downloader.urls_file):
            print(f"\n📁 File ditemukan: {downloader.urls_file}")
            
            choice = input("Jalankan batch download? (y/n): ").lower()
            if choice == 'y':
                downloader.batch_download()
            else:
                interactive_mode(downloader)
        else:
            print(f"\n⚠️  File tidak ditemukan: {downloader.urls_file}")
            print("Membuat file urls.txt...")
            
            # Buat file template
            with open(downloader.urls_file, 'w', encoding='utf-8') as f:
                f.write("# Masukkan URL YouTube (satu per baris)\n")
                f.write("# Contoh:\n")
                f.write("# https://www.youtube.com/watch?v=VIDEO_ID\n\n")
            
            print(f"✅ File dibuat: {downloader.urls_file}")
            print("Tambahkan URL, lalu jalankan program lagi.")
            
            interactive_mode(downloader)
    
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if downloader.config.get('debug_mode'):
            traceback.print_exc()
    
    finally:
        downloader._cleanup_resources()
        downloader.printer.stop()
        input("\nTekan Enter untuk keluar...")


if __name__ == "__main__":
    main()