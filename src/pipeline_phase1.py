import os
import sys
import argparse
from datetime import datetime

# Tambahkan path untuk import module kita
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_extractor import extract_audio_from_video, get_video_info
from src.transcriber import PodcastTranscriber

def phase1_pipeline(video_path, model_size="base", output_dir="data/processed"):
    """
    Pipeline lengkap Fase 1: Video → Audio → Transkripsi
    
    Args:
        video_path (str): Path ke file video
        model_size (str): Ukuran model Whisper
        output_dir (str): Folder untuk output
    
    Returns:
        dict: Hasil dan metadata proses
    """
    print("="*60)
    print("FAZE 1: VIDEO → AUDIO → TRANSCRIPTION")
    print("="*60)
    
    results = {
        "video_file": os.path.basename(video_path),
        "timestamp": datetime.now().isoformat(),
        "model_used": model_size,
        "steps": {}
    }
    
    # Step 1: Cek file video
    if not os.path.exists(video_path):
        print(f"❌ File tidak ditemukan: {video_path}")
        return None
    
    video_info = get_video_info(video_path)
    if video_info:
        print(f"📹 Video: {video_info['filename']}")
        print(f"   ⏱️  Durasi: {video_info['duration']:.2f} detik")
        print(f"   🖼️  Resolusi: {video_info['size'][0]}x{video_info['size'][1]}")
        results["steps"]["video_info"] = video_info
    
    # Step 2: Ekstrak audio
    print("\n" + "="*40)
    print("STEP 1: EKSTRAKSI AUDIO")
    print("="*40)
    
    # Buat nama file audio
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_output = os.path.join(output_dir, f"{base_name}_audio.mp3")
    os.makedirs(os.path.dirname(audio_output), exist_ok=True)
    
    audio_file = extract_audio_from_video(video_path, audio_output)
    
    if not audio_file:
        print("❌ Gagal mengekstrak audio")
        return None
    
    results["steps"]["audio_extraction"] = {
        "audio_file": audio_file,
        "success": True
    }
    
    # Step 3: Transkripsi
    print("\n" + "="*40)
    print("STEP 2: TRANSCRIPTION")
    print("="*40)
    
    try:
        transcriber = PodcastTranscriber(model_size=model_size)
        transcription = transcriber.transcribe_audio(audio_file)
        
        # Simpan hasil
        json_path, srt_path, txt_path = transcriber.save_transcription(
            transcription, output_dir
        )
        
        results["steps"]["transcription"] = {
            "success": True,
            "json_file": json_path,
            "srt_file": srt_path,
            "txt_file": txt_path,
            "num_segments": len(transcription["segments"]),
            "processing_time": transcription["processing_time_seconds"]
        }
        
        # Tampilkan summary
        print("\n" + "="*60)
        print("✅ FAZE 1 SELESAI!")
        print("="*60)
        print(f"📁 Hasil disimpan di folder: {output_dir}")
        print(f"📄 File yang dihasilkan:")
        print(f"   🔊 Audio: {os.path.basename(audio_file)}")
        print(f"   📋 Transkripsi: {os.path.basename(json_path)}")
        print(f"   🎬 Subtitle: {os.path.basename(srt_path)}")
        print(f"   📝 Text: {os.path.basename(txt_path)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error dalam transkripsi: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Pipeline Fase 1: Ekstraksi Audio dan Transkripsi")
    parser.add_argument("--video", "-v", help="Path ke file video", required=False)
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Ukuran model Whisper (default: base)")
    parser.add_argument("--output", "-o", default="data/processed",
                       help="Folder output (default: data/processed)")
    
    args = parser.parse_args()
    
    # Jika video tidak diberikan, coba cari sample
    if not args.video:
        # Cek file sample
        sample_video = "samples/sample_video.mp4"
        if os.path.exists(sample_video):
            args.video = sample_video
            print(f"ℹ️  Menggunakan file sample: {sample_video}")
        else:
            print("❌ File video tidak diberikan dan sample tidak ditemukan")
            print("💡 Cara penggunaan:")
            print("   python pipeline_phase1.py --video path/to/video.mp4")
            print("   python pipeline_phase1.py --video path/to/video.mp4 --model small")
            return
    
    # Jalankan pipeline
    results = phase1_pipeline(args.video, args.model, args.output)
    
    if results:
        # Simpan log
        log_file = os.path.join(args.output, "pipeline_log.json")
        import json
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.write("\n")
        
        print(f"\n📊 Log disimpan: {log_file}")

if __name__ == "__main__":
    main()