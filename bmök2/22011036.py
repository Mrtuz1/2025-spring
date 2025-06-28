import numpy as np
import urllib.request
import tarfile
import os
import pickle
from PIL import Image  # yalnızca görselleştirme için izinli
import matplotlib.pyplot as plt  # yalnızca görselleştirme için izinli

# Kodun çalışabilmesi için CIFAR-10 veri setinin kod ile aynı dizinde olması gerekiyor.

def extract_cifar10():
    filename = "cifar-10-python.tar.gz"
    if os.path.exists(filename):
        print("CIFAR-10 veri seti çıkarılıyor...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print("Çıkarma tamamlandı!")
    else:
        print("CIFAR-10 veri seti bulunamadı!")

def load_cifar10_data():
    data_dir = "cifar-10-batches-py"
    target_classes = ['bird', 'cat', 'dog', 'frog', 'horse']

    class_indices = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    target_indices = [idx for idx, name in class_indices.items() if name in target_classes]

    all_images = []
    all_labels = []

    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']
            all_images.extend(images)
            all_labels.extend(labels)

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    return all_images, all_labels, target_classes, target_indices

def prepare_data():
    extract_cifar10()
    all_images, all_labels, target_classes, target_indices = load_cifar10_data()

    X_train = [] # eğitim verileri
    y_train = [] # sınıf etiketleri
    X_test = [] # test verileri
    y_test = [] # sınıf etiketleri

    for class_idx in target_indices:
        class_sample_indices = np.where(all_labels == class_idx)[0]

        if len(class_sample_indices) >= 55:
            train_indices = class_sample_indices[:50]
            test_indices = class_sample_indices[50:55]

            X_train.extend(all_images[train_indices])
            y_train.extend(all_labels[train_indices])
            X_test.extend(all_images[test_indices])
            y_test.extend(all_labels[test_indices])
        else:
            print(f"Uyarı: {target_classes[target_indices.index(class_idx)]} sınıfı için yeterli örnek yok!")

    # Etiketleri 0-4 aralığında sırala
    label_map = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4}
    y_train = np.array([label_map[label] for label in y_train]) # numpy array'e çevirir ve daha verimli hale getirir
    y_test = np.array([label_map[label] for label in y_test])

    return np.array(X_train), y_train, np.array(X_test), y_test, target_classes

def convert_image(flat_image):
    # 3072 elemanlı CIFAR-10 resmini (R,G,B ayrı düzlemler) 32x32x3 formatına dönüştürür.
    r = flat_image[:1024].reshape((32, 32))
    g = flat_image[1024:2048].reshape((32, 32))
    b = flat_image[2048:].reshape((32, 32))

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def calculate_histogram(channel_data):
    """
    Verilen 2D kanal verisi için 256-bin histogram döndürür.
    """
    hist = np.zeros(256, dtype=np.float32)
    for i in range(channel_data.shape[0]):
        for j in range(channel_data.shape[1]):
            pixel_value = channel_data[i, j]
            hist[pixel_value] += 1
    return hist

def normalized_histograms(images):
    """
    Her resim için RGB histogramlarını hesaplar ve normalize eder.
    
    Args:
        images: (n_samples, 3072) boyutunda resim dizisi
    
    Returns:
        histograms: (n_samples, 3, 256) boyutunda normalize edilmiş histogram dizisi
    """
    n_samples = images.shape[0]
    histograms = np.zeros((n_samples, 3, 256))  # RGB 256-bin

    for i in range(n_samples):
        img = convert_image(images[i])
        for channel in range(3):
            hist = calculate_histogram(img[:, :, channel])
            hist /= (32 * 32)  # normalize et
            histograms[i, channel] = hist

    return histograms

def euclidean_distance(hist1, hist2):
    """
    İki (3, 256) boyutlu normalize RGB histogramı arasındaki Öklid mesafesini hesaplar.
    
    Args:
        hist1: np.array, şekli (3, 256)
        hist2: np.array, şekli (3, 256)

    Returns:
        float: Öklid mesafesi
    """
    distance = 0.0
    for channel in range(3):  # R, G, B
        diff = hist1[channel] - hist2[channel]
        distance += np.sqrt(np.sum(diff ** 2))
    return distance

def selection_sort(distances):
    """
    (indeks, mesafe) tuple'larından oluşan listeyi mesafelere göre küçükten büyüğe sıralar.
    """
    n = len(distances)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if distances[j][1] < distances[min_idx][1]:
                min_idx = j
        # Yer değiştir
        distances[i], distances[min_idx] = distances[min_idx], distances[i]
    return distances

def find_top5_nearest_neighbors(test_hist, train_hists):
    """
    Bir test histogramına en yakın 5 eğitim örneğini bulur.

    Args:
        test_hist: (3, 256) normalize histogram
        train_hists: (n_train, 3, 256)

    Returns:
        top5_indices: en yakın 5 eğitim resminin indeksleri
    """
    distances = []
    for i in range(train_hists.shape[0]):
        dist = euclidean_distance(test_hist, train_hists[i])
        distances.append((i, dist))

    sorted_distances = selection_sort(distances)
    top5_indices = [idx for idx, _ in sorted_distances[:5]]
    return top5_indices


def show_sample_images(X, y, class_names, num_samples=5):
    for class_idx in range(len(class_names)):
        indices = np.where(y == class_idx)[0][:num_samples]
        plt.figure(figsize=(10, 2))
        for i, idx in enumerate(indices):
            img = convert_image(X[idx])
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.suptitle(f"Sınıf: {class_names[class_idx]}")
        plt.show()

def visualize_nearest_neighbors(test_idx, X_test, y_test, test_histograms, X_train, train_histograms, y_train, class_names):
    # Test resmi ve en benzer 5 eğitim resmi indeksini bul
    top5_indices = find_top5_nearest_neighbors(test_histograms[test_idx], train_histograms)

    plt.figure(figsize=(15, 3))

    # Test resmini göster (ilk görsel)
    plt.subplot(1, 6, 1)
    plt.imshow(convert_image(X_test[test_idx]))
    plt.title(f"Test Resmi\n({class_names[y_test[test_idx]]})")
    plt.axis('off')

    # 5 komşuyu sırayla göster
    for i, idx in enumerate(top5_indices):
        plt.subplot(1, 6, i + 2)
        plt.imshow(convert_image(X_train[idx]))
        plt.title(f"Benzer {i+1}\n({class_names[y_train[idx]]})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train, X_test, y_test, class_names = prepare_data()

    print("\nVeri seti başarıyla hazırlandı!")
    print(f"Eğitim veri seti boyutu: {X_train.shape}")
    print(f"Test veri seti boyutu: {X_test.shape}")
    print(f"Sınıflar: {class_names}")

    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {np.sum(y_train == i)} eğitim, {np.sum(y_test == i)} test örneği")

    # RGB histogramlarını hesapla
    print("\nRGB histogramları hesaplanıyor...")
    train_histograms = normalized_histograms(X_train)
    test_histograms = normalized_histograms(X_test)
    
    print(f"Eğitim histogramları boyutu: {train_histograms.shape}")
    print(f"Test histogramları boyutu: {test_histograms.shape}")
    
    # Tüm test resimleri için en yakın komşuları göster
    print("\nTüm test resimleri için en yakın komşular görselleştiriliyor...")
    for test_idx in range(len(X_test)):
        print(f"\nTest resmi {test_idx + 1}/{len(X_test)} - Sınıf: {class_names[y_test[test_idx]]}")
        visualize_nearest_neighbors(test_idx, X_test, y_test, test_histograms, X_train, train_histograms, y_train, class_names)
        

if __name__ == "__main__":
    main()
