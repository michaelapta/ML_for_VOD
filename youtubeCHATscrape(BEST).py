import pandas as pd
import re
import time
from urllib.parse import parse_qs, urlparse

# Pustaka Sastrawi untuk Preprocessing
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Pustaka Scraping & Progress Bar
import pytchat
from tqdm import tqdm

# ==========================================
# 1. INISIALISASI PREPROCESSING
# ==========================================
print("⏳ Memuat pustaka Sastrawi...")
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text) 
    text = re.sub(r'[^a-z\s]', ' ', text)      
    text = re.sub(r'\s+', ' ', text).strip()   
    text = stopword_remover.remove(text)
    return text

tqdm.pandas(desc="🧹 Membersihkan Teks")

# ==========================================
# 2. FUNGSI SCRAPING CEPAT (HIGH SPEED)
# ==========================================
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"youtu.be"}: return parsed.path.strip("/")
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]: return query["v"][0]
        match = re.search(r"/(?:shorts|live)/([A-Za-z0-9_-]{6,})", parsed.path)
        if match: return match.group(1)
    raise ValueError("URL YouTube tidak valid.")

def scrape_vod_chat_fast(url: str, max_items: int = 100, start_time: int = 0) -> list[dict]:
    video_id = extract_video_id(url)
    chat = pytchat.create(video_id=video_id, seektime=start_time)
    comments = []
    
    print(f"📥 Memulai Scraping Cepat (Tanpa Delay) ID: {video_id}...")
    
    with tqdm(total=max_items, desc="📥 Progress Scraping", unit=" chat") as pbar:
        while len(comments) < max_items:
            if not chat.is_alive():
                break
            
            data = chat.get()
            # MENGGUNAKAN AKSES LANGSUNG .items UNTUK KECEPATAN (FIX PYLANCE)
            items = getattr(data, 'items', [])
            
            if not items:
                # Jika data kosong, tunggu sebentar saja (0.1 detik) agar tidak membebani CPU
                time.sleep(0.1)
                continue

            for item in items:
                message = str(getattr(item, "message", "")).strip()
                timestamp = str(getattr(item, "elapsedTime", "")).strip() 
                
                if timestamp.startswith("-"): continue
                
                if message:
                    comments.append({
                        "timestamp": timestamp, 
                        "text": message,
                        "label": 0
                    })
                    pbar.update(1)
                
                if len(comments) >= max_items:
                    break
                    
    chat.terminate()
    return comments

# ==========================================
# 3. EKSEKUSI PROGRAM UTAMA
# ==========================================
if __name__ == "__main__":
    
    URL_TARGET = "https://www.youtube.com/live/ERp8utkZN80?si=OWE4bOBLhRy12aCA" 
    JUMLAH_DATA = 5500  # Sekarang bisa 500+ dengan cepat
    WAKTU_MULAI_DETIK = 0  # Mulai dari menit ke-58:45 (3525 detik) 
    
    try:
        # 1. Scraping (Fast Mode)
        raw_list = scrape_vod_chat_fast(URL_TARGET, JUMLAH_DATA, WAKTU_MULAI_DETIK)
            
        # 2. DataFrame Raw
        df_raw = pd.DataFrame(raw_list)
        df_raw = df_raw[['timestamp', 'text', 'label']]
        
        # 3. DataFrame Clean
        df_clean = df_raw.copy()
        df_clean['text'] = df_clean['text'].progress_apply(preprocess_text) 
        
        # 4. Simpan ke CSV
        df_raw.to_csv("vod_raw.csv", index=False, encoding='utf-8')
        df_clean.to_csv("vod_clean.csv", index=False, encoding='utf-8')
        
        print("\n" + "=" * 50)
        print("🎉 PROSES SELESAI!")
        print(f"✅ Berhasil mengambil {len(df_raw)} chat.")
        print("📂 File: vod_raw.csv & vod_clean.csv")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Terjadi kesalahan: {e}")