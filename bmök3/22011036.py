import hashlib
import sys
import time
import random
import string

# ----------------------------------------------------------
# 🔧 sha256_hex(data: bytes) -> str
# Verilen veriyi SHA-256 hash'ler ve hexadecimal (16'lık sistemde) bir string olarak döner.
# Python'un hashlib modülünü kullanır.
# ----------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ----------------------------------------------------------
# 🔧 count_leading_zero_bits(hex_str) -> int
# Verilen hexadecimal hash string'ini ikilik (binary) hale çevirir
# ve başında kaç tane '0' biti olduğunu sayar.
# PoW zorluğunu kontrol etmek için gereklidir.
# ----------------------------------------------------------
def count_leading_zero_bits(hex_str):
    bin_str = bin(int(hex_str, 16))[2:].zfill(256)  # 256-bit uzunluğa sıfırla
    return len(bin_str) - len(bin_str.lstrip('0'))

# ----------------------------------------------------------
# 🔧 generate_random_prefix(length=4) -> str
# Belirtilen uzunlukta rastgele bir 7-bit ASCII string (prefix) üretir.
# Karakterler sadece harfler ve sayılardan oluşur.
# ----------------------------------------------------------
def generate_random_prefix(length=4):
    chars = string.ascii_letters + string.digits  # a-zA-Z0-9
    return ''.join(random.choice(chars) for _ in range(length))

# ----------------------------------------------------------
# 🔧 sha256_of_file(filename) -> str
# Dosya içeriğini okur, SHA-256 hash'ini hesaplayıp hexadecimal string olarak döner.
# ----------------------------------------------------------
def sha256_of_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    return sha256_hex(data)

# ----------------------------------------------------------
# 🔧 parse_header(file_path) -> dict
# pow-create tarafından üretilen powheader.txt içindeki başlıkları okuyup
# key:value şeklinde sözlüğe (dict) çevirir.
# ----------------------------------------------------------
def parse_header(file_path):
    headers = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
    return headers

# ----------------------------------------------------------
# 🧠 pow_create(nbits: int, filename: str)
# Verilen dosya için Proof-of-Work üretir.
# - SHA-256 hash hesaplar
# - Random W prefix'leri dener
# - W + hash sonucu tekrar SHA-256 ile hash'lenir
# - İstenilen sayıda sıfır bit (leading zeros) elde edilene kadar devam eder
# - Sonuçları hem ekrana hem dosyaya yazdırır
# ----------------------------------------------------------
def pow_create(nbits, filename):
    file_hash = sha256_of_file(filename)
    iterations = 0
    start_time = time.time()

    while True:
        prefix = generate_random_prefix(4 + iterations // 1_000_000)
        combined = (prefix + file_hash).encode('utf-8')
        pow_hash = sha256_hex(combined)
        leading = count_leading_zero_bits(pow_hash)
        iterations += 1
        if leading >= nbits:
            break

    elapsed = time.time() - start_time

    # Ekrana yazdır
    print(f"File: {filename}")
    print(f"Initial-hash: {file_hash}")
    print(f"Proof-of-work: {prefix}")
    print(f"Hash: {pow_hash}")
    print(f"Leading-bits: {leading}")
    print(f"Iterations: {iterations}")
    print(f"Compute-time: {elapsed:.5f}")

    # Dosyaya başlıkları yaz
    with open("powheader.txt", "w") as out:
        out.write(f"File: {filename}\n")
        out.write(f"Initial-hash: {file_hash}\n")
        out.write(f"Proof-of-work: {prefix}\n")
        out.write(f"Hash: {pow_hash}\n")
        out.write(f"Leading-bits: {leading}\n")
        out.write(f"Iterations: {iterations}\n")
        out.write(f"Compute-time: {elapsed:.5f}\n")

    print("PoW başarıyla dosyaya yazıldı: powheader.txt")

# ----------------------------------------------------------
# 🧠 pow_check(header_filename: str, original_filename: str)
# pow-create tarafından üretilmiş bir başlık dosyasını kontrol eder:
# - Initial hash dosya içeriğiyle aynı mı?
# - Proof-of-work string + hash sonucu doğru hash’i üretmiş mi?
# - Hash’in başındaki sıfır bit sayısı, belirtilen değere eşit mi?
# ----------------------------------------------------------
def pow_check(header_filename, original_filename):
    headers = parse_header(header_filename)

    try:
        initial_hash_expected = headers["Initial-hash"]
        proof_of_work = headers["Proof-of-work"]
        pow_hash_expected = headers["Hash"]
        leading_bits_expected = int(headers["Leading-bits"])
    except KeyError as e:
        print(f"Başlık eksik: {e}")
        sys.exit(1)

    actual_initial_hash = sha256_of_file(original_filename)
    if actual_initial_hash != initial_hash_expected:
        print("HATA: Initial-hash değeri dosya içeriğiyle eşleşmiyor.")
        return

    combined = (proof_of_work + initial_hash_expected).encode('utf-8')
    actual_pow_hash = sha256_hex(combined)
    if actual_pow_hash != pow_hash_expected:
        print("HATA: Proof-of-work hash sonucu eşleşmiyor.")
        return

    actual_leading_bits = count_leading_zero_bits(actual_pow_hash)
    if actual_leading_bits != leading_bits_expected:
        print(f"HATA: Leading-bits ({actual_leading_bits}) değeri, başlıkta belirtilen ({leading_bits_expected}) ile uyuşmuyor.")
        return

    print("başarılı")

# ----------------------------------------------------------
# 🚀 main()
# Komut satırından çağrıldığında hangi işlemin yapılacağını belirler:
# - pow.py create 20 walrus.txt
# - pow.py check powheader.txt walrus.txt
# ----------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python pow.py create <nbits> <filename>")
        print("  python pow.py check <headerfile> <filename>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        if len(sys.argv) != 4:
            print("Kullanım: python pow.py create <nbits> <filename>")
            sys.exit(1)
        nbits = int(sys.argv[2])
        filename = sys.argv[3]
        pow_create(nbits, filename)

    elif command == "check":
        if len(sys.argv) != 4:
            print("Kullanım: python pow.py check <headerfile> <filename>")
            sys.exit(1)
        header_filename = sys.argv[2]
        original_filename = sys.argv[3]
        pow_check(header_filename, original_filename)

    else:
        print("Geçersiz komut. 'create' ya da 'check' kullanılmalı.")
        sys.exit(1)

if __name__ == "__main__":
    main()
