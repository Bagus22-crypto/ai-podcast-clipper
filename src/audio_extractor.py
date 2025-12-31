import os
from moviepy.editor import VideoFileClip
import warnings
warnings.filterwarnings('ignore')

def extract_audio_from_video(video_path, audio_output_path=None):
    """
    Ekstrak audio dari file video dan simpan sebagai MP3.
    
    Args:
        video_path (str): Path ke file video
        audio_output_path (str): Path output audio (optional)
    
    Returns:
        str: Path ke file audio yang dihasilkan
    """
    try:
        print(f"📹 Membaca video: {os.path.basename(video_path)}")
        
        # Jika output path tidak diberikan, buat otomatis
        if audio_output_path is None:
            base_name = os.path.splitext(video_path)[0]
            audio_output_path = f"{base_name}_audio.mp3"
        
        # Load video dan ekstrak audio
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Simpan sebagai MP3
        print(f"🔊 Mengekstrak audio...")
        audio.write_audiofile(audio_output_path, verbose=False, logger=None)
        
        # Tutup file untuk menghemat memory
        audio.close()
        video.close()
        
        print(f"✅ Audio berhasil disimpan: {audio_output_path}")
        print(f"   Durasi: {video.duration:.2f} detik")
        
        return audio_output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_video_info(video_path):
    """Dapatkan informasi dasar video"""
    try:
        video = VideoFileClip(video_path)
        info = {
            'duration': video.duration,
            'fps': video.fps,
            'size': video.size,
            'filename': os.path.basename(video_path)
        }
        video.close()
        return info
    except:
        return None

# Contoh penggunaan
if __name__ == "__main__":
    # Test dengan file video (jika ada)
    test_video = "samples/sample_video.mp4"
    
    if os.path.exists(test_video):
        info = get_video_info(test_video)
        if info:
            print(f"Video Info:")
            print(f"  Nama: {info['filename']}")
            print(f"  Durasi: {info['duration']:.2f} detik")
            print(f"  Resolusi: {info['size']}")
            print(f"  FPS: {info['fps']}")
        
        audio_file = extract_audio_from_video(test_video)
    else:
        print("⚠️  File sample_video.mp4 tidak ditemukan di folder 'samples'")
        print("💡 Silakan tambahkan file video pendek untuk testing")