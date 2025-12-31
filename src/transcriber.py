import whisper
import json
import os
import time
from datetime import timedelta

class PodcastTranscriber:
    def __init__(self, model_size="base"):
        """
        Inisialisasi transcriber dengan model Whisper.
        
        Args:
            model_size (str): Ukuran model ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"🔄 Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size
        print(f"✅ Model {model_size} siap digunakan")
    
    def transcribe_audio(self, audio_path, language=None):
        """
        Transkripsi audio ke teks dengan timestamp.
        
        Args:
            audio_path (str): Path ke file audio
            language (str): Kode bahasa (optional, auto-detect jika None)
        
        Returns:
            dict: Hasil transkripsi dengan segments
        """
        print(f"🎵 Memproses: {os.path.basename(audio_path)}")
        
        start_time = time.time()
        
        # Options untuk transkripsi
        options = {
            "task": "transcribe",
            "verbose": False,  # Nonaktifkan output verbose untuk clean terminal
            "fp16": False      # Gunakan float32 untuk kompatibilitas lebih baik
        }
        
        if language:
            options["language"] = language
        
        # Transkripsi
        result = self.model.transcribe(audio_path, **options)
        
        processing_time = time.time() - start_time
        
        # Format hasil
        transcription = {
            "audio_file": os.path.basename(audio_path),
            "model_used": self.model_size,
            "processing_time_seconds": round(processing_time, 2),
            "language": result.get("language", "unknown"),
            "full_text": result["text"],
            "segments": []
        }
        
        # Format segments dengan informasi yang lebih detail
        for i, segment in enumerate(result["segments"]):
            transcription["segments"].append({
                "id": i,
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "duration": round(segment["end"] - segment["start"], 2),
                "text": segment["text"].strip(),
                "confidence": round(segment.get("avg_logprob", 0), 3) if segment.get("avg_logprob") else None
            })
        
        print(f"✅ Transkripsi selesai!")
        print(f"   ⏱️  Waktu proses: {processing_time:.2f} detik")
        print(f"   📝 Jumlah segment: {len(transcription['segments'])}")
        print(f"   🔤 Jumlah karakter: {len(transcription['full_text'])}")
        
        return transcription
    
    def save_transcription(self, transcription, output_dir="data/processed"):
        """
        Simpan hasil transkripsi ke berbagai format.
        
        Args:
            transcription (dict): Hasil transkripsi
            output_dir (str): Folder output
        
        Returns:
            tuple: (json_path, srt_path, txt_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(transcription["audio_file"])[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Save as JSON (structured data)
        json_path = os.path.join(output_dir, f"{base_name}_transcript_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        
        # 2. Save as SRT (subtitle format)
        srt_path = os.path.join(output_dir, f"{base_name}_subtitles_{timestamp}.srt")
        self._save_as_srt(transcription["segments"], srt_path)
        
        # 3. Save as TXT (plain text)
        txt_path = os.path.join(output_dir, f"{base_name}_text_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Transkripsi: {transcription['audio_file']}\n")
            f.write(f"Model: {transcription['model_used']}\n")
            f.write(f"Bahasa: {transcription['language']}\n")
            f.write(f"Waktu proses: {transcription['processing_time_seconds']} detik\n")
            f.write("="*50 + "\n\n")
            
            for segment in transcription["segments"]:
                start_str = str(timedelta(seconds=segment["start"])).split(".")[0]
                f.write(f"[{start_str}] {segment['text']}\n")
        
        print(f"💾 Hasil disimpan:")
        print(f"   📄 JSON: {os.path.basename(json_path)}")
        print(f"   📝 SRT: {os.path.basename(srt_path)}")
        print(f"   📋 TXT: {os.path.basename(txt_path)}")
        
        return json_path, srt_path, txt_path
    
    def _save_as_srt(self, segments, output_path):
        """Simpan sebagai format SRT (SubRip Text)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                # Format timestamp SRT
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _format_timestamp(self, seconds):
        """Format detik ke HH:MM:SS,mmm untuk SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

# Contoh penggunaan
if __name__ == "__main__":
    # Test dengan file audio (jika ada)
    test_audio = "samples/sample_audio.mp3"
    
    if os.path.exists(test_audio):
        print("="*50)
        print("TEST TRANSCRIPTION")
        print("="*50)
        
        # Gunakan model kecil untuk testing cepat
        transcriber = PodcastTranscriber(model_size="tiny")
        
        # Transkripsi
        result = transcriber.transcribe_audio(test_audio)
        
        # Simpan hasil
        transcriber.save_transcription(result)
        
        # Tampilkan preview
        print("\n📋 PREVIEW TRANSCRIPT (5 segment pertama):")
        print("-"*40)
        for i, segment in enumerate(result["segments"][:5]):
            start_str = str(timedelta(seconds=segment["start"])).split(".")[0]
            print(f"[{start_str}] {segment['text'][:100]}...")
        
        if len(result["segments"]) > 5:
            print(f"... dan {len(result['segments']) - 5} segment lainnya")
            
    else:
        print("⚠️  File sample_audio.mp3 tidak ditemukan")
        print("💡 Ekstrak audio dari video terlebih dahulu")