import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from deap import base, creator, tools, algorithms
from pathlib import Path

# ========== 1. Resimleri Yükleme ==========  
def load_binary_image(filename, size=24):
    """ Belirtilen PNG dosyasını yükler, 24x24 boyutuna getirir ve binary numpy array olarak döndürür. """
    img = Image.open(filename).convert('L')  # Gri tonlamaya çevir
    img = img.resize((size, size))  # 24x24 boyutuna getir
    img_array = np.array(img)  
    return np.where(img_array > 128, 1, 0)  # Beyaz pikseller (255) -> 1, Siyah pikseller (0) -> 0

# Kodun bulunduğu dizini al
current_dir = Path(__file__).parent

# 5 farklı resmi yükle
image_filenames = [current_dir / "diamond.png", 
                   current_dir / "donut.png", 
                   current_dir / "ghost.png", 
                   current_dir / "glasses.png", 
                   current_dir / "heart.png"]

binary_images = [load_binary_image(img) for img in image_filenames]

# ========== 2. Pattern'lerin Başlangıçta Tanımlanması ==========
def generate_random_patterns(num_patterns=7):
    """ Rastgele 3x3 pattern'ler oluşturur """
    return [np.random.choice([0, 1], size=(3, 3)) for _ in range(num_patterns)]

# 7 adet rastgele pattern başlat
initial_patterns = generate_random_patterns()

# ========== 3. Pattern ile Resim Bloklarını Eşleme ==========
def match_best_pattern(block, patterns):
    """ 3x3'lük bloğa en iyi eşleşen pattern'i bulur """
    min_loss = float('inf')
    best_pattern = None
    for pattern in patterns:
        loss = np.sum(np.abs(block - pattern))  # Hamming mesafesi (farklı piksellerin sayısı)
        if loss < min_loss:
            min_loss = loss
            best_pattern = pattern
    return best_pattern, min_loss

def calculate_total_loss(image, patterns):
    """ Bir resmin toplam loss'unu hesaplar """
    total_loss = 0
    for i in range(0, 24, 3):
        for j in range(0, 24, 3):
            block = image[i:i+3, j:j+3]
            _, loss = match_best_pattern(block, patterns)
            total_loss += loss
    return total_loss

# ========== 4. Genetik Algoritma (GA) için Ayarlar ==========
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Loss'u minimize ediyoruz
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 63)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ========== 5. Fitness Fonksiyonu ==========
def evaluate(individual):
    """ Pattern setini alır ve toplam loss'u hesaplar """
    patterns = [np.array(individual[i*9:(i+1)*9]).reshape((3,3)) for i in range(7)]
    total_loss = sum(calculate_total_loss(img, patterns) for img in binary_images)
    return (total_loss,)

toolbox.register("evaluate", evaluate)

# ========== 6. GA Operatörleri ==========
toolbox.register("mate", tools.cxTwoPoint)  # Çift noktalı çaprazlama
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # %5 ihtimalle bit değişimi
toolbox.register("select", tools.selTournament, tournsize=3)  # Turnuva seçimi

# ========== 7. GA Çalıştırma ==========
def run_ga(n_gen=100, pop_size=50):
    """ Genetik Algoritma'yı çalıştır ve en iyi pattern setini döndür """
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # En iyi pattern seti saklanacak
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)  # En düşük loss takip edilecek
    stats.register("mean", np.mean)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "mean", "max"]
    
    for gen in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        
        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)
        
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        print(logbook.stream)
    
    return hof[0], logbook

best_pattern_set, logbook = run_ga()

# ========== 8. Sonuçları Görselleştirme ==========
def visualize_patterns(patterns, title="Patterns"):
    """ Pattern'leri görselleştirir """
    fig, axs = plt.subplots(1, 7, figsize=(10, 2))
    for i, pattern in enumerate(patterns):
        axs[i].imshow(pattern, cmap='gray')
        axs[i].axis('off')
    plt.suptitle(title)
    plt.show()

# En iyi pattern setini çıkar
best_patterns = [np.array(best_pattern_set[i*9:(i+1)*9]).reshape((3,3)) for i in range(7)]
visualize_patterns(best_patterns, title="En İyi Pattern Seti")

# Yüklenen resimleri görselleştir
fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for i, img in enumerate(binary_images):
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
plt.suptitle("Yüklenen 24x24 Resimler")
plt.show()

# Jenerasyon başına minimum loss grafiği çizdirme
plt.plot(logbook.select("gen"), logbook.select("min"), label="Min Loss")
plt.plot(logbook.select("gen"), logbook.select("mean"), label="Mean Loss")
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Değişimi")
plt.show()

#Mutasyon oranlarını kıyaslama grafiği
def test_mutation_rates(mut_rates, n_gen=100, pop_size=50):
    results = {}
    
    for mutpb in mut_rates:
        print(f"Testing mutation rate: {mutpb}")
        toolbox.register("mutate", tools.mutFlipBit, indpb=mutpb)  # Güncellenmiş mutasyon oranı
        best_ind, logbook = run_ga(n_gen=n_gen, pop_size=pop_size)
        results[mutpb] = logbook.select("min")  # Minimum loss değerlerini sakla
    
    # Sonuçları grafik üzerinde gösterme
    plt.figure(figsize=(10, 5))
    for mutpb, losses in results.items():
        plt.plot(range(n_gen), losses, label=f"Mutation: {mutpb}")
    
    plt.xlabel("Generation")
    plt.ylabel("Min Loss")
    plt.title("Mutasyon Oranlarının Etkisi")
    plt.legend()
    plt.show()

# Test etmek için farklı mutasyon oranları
mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
test_mutation_rates(mutation_rates)

def test_population_sizes(pop_sizes, mutpb=0.05, n_gen=100):
    results = {}
    
    # Mutasyon oranını sabit tut
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutpb)

    for size in pop_sizes:
        print(f"Testing population size: {size}")
        best_ind, logbook = run_ga(n_gen=n_gen, pop_size=size)
        results[size] = logbook.select("min")  # Minimum loss değerlerini kaydet

    # Grafikle sonuçları göster
    plt.figure(figsize=(10, 5))
    for size, losses in results.items():
        plt.plot(range(n_gen), losses, label=f"Population: {size}")
    
    plt.xlabel("Generation")
    plt.ylabel("Min Loss")
    plt.title(f"Popülasyon Boyutunun Etkisi (Mutasyon: {mutpb})")
    plt.legend()
    plt.show()

population_sizes = [20, 50, 100, 200]
test_population_sizes(population_sizes)