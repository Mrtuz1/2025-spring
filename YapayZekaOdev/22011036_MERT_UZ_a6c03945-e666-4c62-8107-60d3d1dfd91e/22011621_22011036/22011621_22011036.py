# Gerekli kütüphaneleri yükle
import pandas as pd  # Veri işleme
import numpy as np  # Matematiksel işlemler
import torch  # Derin öğrenme
from transformers import AutoTokenizer, AutoModel  # Dil modelleri
from tqdm import tqdm  # İlerleme çubuğu
import random
import os
from sklearn.metrics.pairwise import cosine_similarity  # Benzerlik hesaplama

# GPU ayarları
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
assert torch.cuda.is_available(), "GPU bulunamadı! Bu kod GPU gerektiriyor."
device = torch.device('cuda')
print(f"Kullanılan cihaz: {device} - {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()

# Excel dosyasını oku
def read_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"Excel dosyası başarıyla okundu. Toplam {len(df)} satır bulundu.")
        for col in df.columns:
            if df[col].dtype.kind in 'iuf':
                df[col] = df[col].astype(str)
            df[col] = df[col].fillna("")
        return df
    except Exception as e:
        print(f"Excel dosyası okuma hatası: {e}")
        return None

# Metni temizle
def clean_text(text):
    if isinstance(text, (int, float)):
        return str(text)
    elif not isinstance(text, str):
        return ""
    return str(text).strip()

# Dil modelini yükle
def load_model(model_name):
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# Metinleri vektörlere çevir
def batch_embed(texts, tokenizer, model, batch_size=8):
    all_embeddings = []
    cleaned_texts = [clean_text(text) for text in texts]
    embedding_size = model.config.hidden_size
    
    # Boş metinleri bul
    empty_indices = [i for i, text in enumerate(cleaned_texts) if not text]
    if empty_indices:
        print(f"Uyarı: {len(empty_indices)} boş metin var. Bu metinler için sıfır embeddingler kullanılacak.")
    
    # Boş olmayan metinleri ayır
    non_empty_texts = [text for text in cleaned_texts if text]
    non_empty_indices = [i for i, text in enumerate(cleaned_texts) if text]
    
    # Sıfır vektörler oluştur
    for _ in range(len(cleaned_texts)):
        all_embeddings.append(np.zeros(embedding_size))
    
    # Metinleri gruplar halinde işle
    if non_empty_texts:
        batch_embeddings = []
        for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="Embedding"):
            batch_texts = non_empty_texts[i:i+batch_size]
            try:
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
                batch_embeddings.extend(emb)
            except Exception as e:
                print(f"Batch embedding hatası (batch {i}): {str(e)}")
                for _ in range(len(batch_texts)):
                    batch_embeddings.append(np.zeros(embedding_size))
        
        # Embedding'leri yerleştir
        for idx, emb_idx in enumerate(non_empty_indices):
            if idx < len(batch_embeddings):
                all_embeddings[emb_idx] = batch_embeddings[idx]
    return all_embeddings

# Benzerlik analizi yap
def analyze_similarity(matrix):
    n = matrix.shape[0]
    results = np.zeros((n,4), dtype=int)  # [gpt_top1, gpt_top5, deepseek_top1, deepseek_top5]
    print("Benzerlik analizi yapılıyor...")
    
    for i in tqdm(range(n), desc="Similarity Analysis"):
        q = matrix[i,0]
        
        # GPT benzerlik analizi
        sims2 = [(j, cosine_similarity([q],[matrix[j,1]])[0][0]) for j in range(n)]
        sims2.sort(key=lambda x: x[1], reverse=True)
        top5_2 = [j for j,_ in sims2[:5]]
        if top5_2[0]==i: results[i,0]=results[i,1]=1  # GPT top1 ve top5
        elif i in top5_2: results[i,1]=1  # GPT top5
        
        # Deepseek benzerlik analizi
        sims3 = [(j, cosine_similarity([q],[matrix[j,2]])[0][0]) for j in range(n)]
        sims3.sort(key=lambda x: x[1], reverse=True)
        top5_3 = [j for j,_ in sims3[:5]]
        if top5_3[0]==i: results[i,2]=results[i,3]=1  # Deepseek top1 ve top5
        elif i in top5_3: results[i,3]=1  # Deepseek top5
    return results

# Korelasyon hesapla
def compute_correlations(class_labels, top1, top5):
    try:
        if len(set(class_labels)) <= 1:
            print("Uyarı: Tüm sınıf etiketleri aynı, korelasyon hesaplanamaz.")
            return float('nan'), float('nan')
        valid_indices = [i for i in range(len(class_labels)) if not np.isnan(class_labels[i])]
        if len(valid_indices) < 2:
            print("Uyarı: Yeterli geçerli veri yok, korelasyon hesaplanamaz.")
            return float('nan'), float('nan')
        valid_classes = [class_labels[i] for i in valid_indices]
        valid_top1 = [top1[i] for i in valid_indices]
        valid_top5 = [top5[i] for i in valid_indices]
        corr_top1 = np.corrcoef(valid_classes, valid_top1)[0,1]
        corr_top5 = np.corrcoef(valid_classes, valid_top5)[0,1]
        return corr_top1, corr_top5
    except Exception as e:
        print(f"Korelasyon hesaplama hatası: {str(e)}")
        return float('nan'), float('nan')

# Vektör kombinasyonlarını hesapla
def calculate_vector_combinations(matrix):
    """
    Vektör kombinasyonlarını hesaplar:
    s: soru vektörü
    g: GPT yanıtı vektörü
    d: Deepseek yanıtı vektörü
    s-g: soru ve GPT yanıtı farkı
    s-d: soru ve Deepseek yanıtı farkı
    g-d: GPT ve Deepseek yanıtları farkı
    |s-g|: soru ve GPT yanıtı arasındaki mesafe
    |s-d|: soru ve Deepseek yanıtı arasındaki mesafe
    |g-d|: GPT ve Deepseek yanıtları arasındaki mesafe
    |s-g|-|s-d|: mesafe farkı
    """
    print("\nVektör kombinasyonları hesaplanıyor...")
    
    # Temel vektörler
    s = matrix[:, 0]  # soru vektörleri
    g = matrix[:, 1]  # GPT yanıtı vektörleri
    d = matrix[:, 2]  # Deepseek yanıtı vektörleri
    
    # Vektör farkları
    s_g = s - g  # soru - GPT
    s_d = s - d  # soru - Deepseek
    g_d = g - d  # GPT - Deepseek
    
    # Mesafeler (L2 norm)
    s_g_dist = np.linalg.norm(s_g, axis=1)  # |s-g|
    s_d_dist = np.linalg.norm(s_d, axis=1)  # |s-d|
    g_d_dist = np.linalg.norm(g_d, axis=1)  # |g-d|
    
    # Mesafe farkları
    s_g_s_d_diff = s_g_dist - s_d_dist  # |s-g| - |s-d|
    
    # Tüm vektörleri birleştir
    vectors = {
        's': s,
        'g': g,
        'd': d,
        's-g': s_g,
        's-d': s_d,
        'g-d': g_d,
        '|s-g|': s_g_dist,
        '|s-d|': s_d_dist,
        '|g-d|': g_d_dist,
        '|s-g|-|s-d|': s_g_s_d_diff
    }
    
    # Vektör boyutlarını kontrol et
    print("\nVektör Boyutları:")
    for name, vec in vectors.items():
        if len(vec.shape) > 1:
            print(f"{name}: {vec.shape}")
        else:
            print(f"{name}: {len(vec)}")
    
    return vectors

# Modelleri eğit ve değerlendir
def train_and_evaluate_models(vectors, class_labels, test_size=0.2, random_state=42):
    """
    Vektör kombinasyonlarını kullanarak modelleri eğitir ve değerlendirir.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    
    print("\nModel eğitimi ve değerlendirmesi başlıyor...")
    
    # Her vektör kombinasyonu için model eğitimi ve değerlendirmesi
    for vector_name, vector_data in vectors.items():
        print(f"\n--- {vector_name} vektörü için model sonuçları ---")
        
        # Vektör verilerini düzleştir (eğer 2D ise)
        if len(vector_data.shape) > 1:
            X = vector_data.reshape(vector_data.shape[0], -1)
        else:
            X = vector_data.reshape(-1, 1)
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, class_labels, test_size=test_size, random_state=random_state, stratify=class_labels
        )
        
        # Modelleri tanımla
        models = {
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state)
        }
        
        # Her model için eğitim ve değerlendirme
        for model_name, model in models.items():
            print(f"\n{model_name} modeli eğitiliyor...")
            model.fit(X_train, y_train)
            
            # Test seti üzerinde tahmin yap
            y_pred = model.predict(X_test)
            
            # Performans metriklerini hesapla
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test seti doğruluk oranı: {accuracy:.4f}")
            print("\nSınıflandırma raporu:")
            print(classification_report(y_test, y_pred, zero_division=0))

# Ana program
def main():
    df = read_excel_file("veri.xlsx")
    if df is None:
        return
    print("\nVeri tipleri kontrolü:")
    for col in df.columns:
        print(f"Sütun '{col}': {df[col].dtype}")
    
    # Kullanılacak modeller
    model_names = [
        "intfloat/multilingual-e5-large-instruct",
        "ytu-ce-cosmos/turkish-e5-large"
    ]
    model_display_names = {
        "intfloat/multilingual-e5-large-instruct": "E5-multilingual",
        "ytu-ce-cosmos/turkish-e5-large": "CosmosE5-Turkish"
    }
    
    # Matris dosyalarının varlığını kontrol et
    matrix_files_exist = all(os.path.exists(f"full_matrix_{i+1}.npy") for i in range(len(model_names)))
    # Topinfo dosyalarının varlığını kontrol et
    topinfo_files_exist = all(os.path.exists(f"matrix{i+1}_topinfo.npy") for i in range(len(model_names)))
    
    if not matrix_files_exist:
        # Matris dosyaları yoksa, modelleri yükle ve yeni matrisler oluştur
        tokenizers, models = {}, {}
        for model_name in model_names:
            try:
                print(f"\n{model_display_names[model_name]} modeli yükleniyor...")
                tokenizers[model_name], models[model_name] = load_model(model_name)
                print(f"{model_display_names[model_name]} modeli başarıyla yüklendi.")
            except Exception as e:
                print(f"Model yükleme hatası ({model_name}): {str(e)}")
        loaded_models = [name for name in model_names if name in models and name in tokenizers]
        if not loaded_models:
            print("Hiçbir model yüklenemedi! Program sonlandırılıyor.")
            return
        
        for model_name in loaded_models:
            print(f"\n{model_display_names[model_name]} modeli için embeddingler oluşturuluyor...")
            embedding_dim = models[model_name].config.hidden_size
            n_samples = len(df)
            matrix = np.zeros((n_samples, 4, embedding_dim))
            for col_idx in range(min(3, len(df.columns)-1)):  # Sadece ilk 3 sütun embedlenecek
                try:
                    col_name = df.columns[col_idx]
                    print(f"Sütun {col_idx+1} ({col_name}) için embeddingler oluşturuluyor...")
                    col_embeddings = batch_embed(df[col_name].tolist(), tokenizers[model_name], models[model_name])
                    for row_idx in range(n_samples):
                        matrix[row_idx, col_idx] = col_embeddings[row_idx]
                except Exception as e:
                    print(f"Sütun {col_idx+1} için embedding oluşturma hatası: {str(e)}")
            
            try:
                last_col_name = df.columns[-1]
                print(f"Son sütun ({last_col_name}) integer olarak ekleniyor...")
                last_column_values = df[last_col_name].astype(str).str.extract('(\d+)').fillna(0).astype(int)[0].to_numpy()
                for row_idx in range(n_samples):
                    matrix[row_idx, 3, 0] = last_column_values[row_idx]
            except Exception as e:
                print(f"Son sütun integer ekleme hatası: {str(e)}")
            
            model_idx = model_names.index(model_name) + 1
            np.save(f"full_matrix_{model_idx}.npy", matrix)
            print(f"{model_display_names[model_name]} için tüm veri embeddingleri kaydedildi: full_matrix_{model_idx}.npy")
    else:
        print("\nMevcut matris dosyaları bulundu. Yeni embedding oluşturulmayacak.")
    
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n--- {model_display_names[model_name]} için Analiz ---")
        try:
            file_path = f"full_matrix_{model_idx}.npy"
            print(f"{file_path} yükleniyor...")
            full_matrix = np.load(file_path)
            n_total = full_matrix.shape[0]
            print(f"Matris yüklendi: {n_total} örnek")
        except Exception as e:
            print(f"Matris yükleme hatası: {str(e)}")
            continue
        
        n_samples = min(1000, n_total)
        sample_indices = random.sample(range(n_total), n_samples) if n_total > n_samples else list(range(n_total))
        matrix = full_matrix[sample_indices]
        
        # Vektör kombinasyonlarını hesapla
        vectors = calculate_vector_combinations(matrix)
        
        # Class etiketlerini al
        class_labels = matrix[:, 3, 0]
        
        # Model eğitimi ve değerlendirmesi
        train_and_evaluate_models(vectors, class_labels)
        
        try:
            if not topinfo_files_exist:
                print("Topinfo dosyaları bulunamadı. Yeni analiz yapılıyor...")
                results = analyze_similarity(matrix)
                np.save(f"matrix{model_idx}_topinfo.npy", results)
            else:
                print("Mevcut topinfo dosyası kullanılıyor...")
                results = np.load(f"matrix{model_idx}_topinfo.npy")
            
            # GPT sonuçları
            gpt_top1 = results[:, 0]
            gpt_top5 = results[:, 1]
            print(f"\n--- {model_display_names[model_name]} GPT Başarıları ---")
            print(f"GPT Top-1 Başarı:  {gpt_top1.mean()*100:.2f}%")
            print(f"GPT Top-5 Başarı:  {gpt_top5.mean()*100:.2f}%")
            
            # Deepseek sonuçları
            deepseek_top1 = results[:, 2]
            deepseek_top5 = results[:, 3]
            print(f"\n--- {model_display_names[model_name]} Deepseek Başarıları ---")
            print(f"Deepseek Top-1 Başarı:  {deepseek_top1.mean()*100:.2f}%")
            print(f"Deepseek Top-5 Başarı:  {deepseek_top5.mean()*100:.2f}%")
            
            class_labels = matrix[:, 3, 0]
            
            # GPT korelasyonları
            gpt_corr_top1, gpt_corr_top5 = compute_correlations(class_labels, gpt_top1, gpt_top5)
            print("\nGPT Örnek Bazında Korelasyon (Pearson r):")
            print(f"  Sınıf vs Top-1: r = {gpt_corr_top1:.3f}")
            print(f"  Sınıf vs Top-5: r = {gpt_corr_top5:.3f}")
            
            # Deepseek korelasyonları
            deepseek_corr_top1, deepseek_corr_top5 = compute_correlations(class_labels, deepseek_top1, deepseek_top5)
            print("\nDeepseek Örnek Bazında Korelasyon (Pearson r):")
            print(f"  Sınıf vs Top-1: r = {deepseek_corr_top1:.3f}")
            print(f"  Sınıf vs Top-5: r = {deepseek_corr_top5:.3f}")
            
            # GPT sınıf bazlı sonuçlar
            df_gpt = pd.DataFrame({
                'class': class_labels,
                'top1': gpt_top1,
                'top5': gpt_top5
            })
            if 0 in df_gpt['class'].values:
                print(f"\nNot: {sum(df_gpt['class'] == 0)} adet sınıfı 0 olan örnek var.")
                df_gpt_class = df_gpt[df_gpt['class'] > 0]
            else:
                df_gpt_class = df_gpt
            
            if not df_gpt_class.empty:
                gpt_summary = df_gpt_class.groupby('class').mean().reset_index()
                print("\nGPT Sınıf Bazlı Ortalama Başarılar:")
                for _, row in gpt_summary.iterrows():
                    print(f"  Class {int(row['class'])}: Top-1 = {row['top1']*100:.2f}%, Top-5 = {row['top5']*100:.2f}%")
            else:
                print("\nUyarı: Hiç geçerli GPT sınıf bulunamadı.")
            
            # Deepseek sınıf bazlı sonuçlar
            df_deepseek = pd.DataFrame({
                'class': class_labels,
                'top1': deepseek_top1,
                'top5': deepseek_top5
            })
            if 0 in df_deepseek['class'].values:
                df_deepseek_class = df_deepseek[df_deepseek['class'] > 0]
            else:
                df_deepseek_class = df_deepseek
            
            if not df_deepseek_class.empty:
                deepseek_summary = df_deepseek_class.groupby('class').mean().reset_index()
                print("\nDeepseek Sınıf Bazlı Ortalama Başarılar:")
                for _, row in deepseek_summary.iterrows():
                    print(f"  Class {int(row['class'])}: Top-1 = {row['top1']*100:.2f}%, Top-5 = {row['top5']*100:.2f}%")
            else:
                print("\nUyarı: Hiç geçerli Deepseek sınıf bulunamadı.")
        except Exception as e:
            print(f"Benzerlik analizi hatası: {str(e)}")

if __name__ == "__main__":
    main()
