# NeuraLite - Minimalist Neural Language Model (SLM) in Pure NumPy

![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

NeuraLite adalah proyek Small Language Model (SLM) yang sepenuhnya diimplementasikan menggunakan Python standard library dan NumPy. Dirancang untuk berjalan di Termux/Android, Google Cloud Shell, atau lingkungan Linux ringan lainnya.

## ✨ Kemampuan Utama

- 🚀 **Pelatihan dari Nol**: Melatih model Transformer decoder-only ringan
- 💬 **Inferensi Interaktif**: Antarmuka CLI berwarna untuk generasi teks
- 📱 **Ringan & Portabel**: Berjalan di perangkat dengan spesifikasi minimal
- 🎯 **Pure NumPy**: Tidak bergantung pada framework ML berat

## 📚 Fitur Lengkap

### 1. Training Script (`training.py`)

- **Dataset Handling**: Membaca semua file `*.txt` secara otomatis
- **Preprocessing**: 
  - Lowercase normalization
  - Regex cleanup
  - Whitespace normalization
- **Tokenization**: Character-level atau simplified subword (BPE-like)
- **Vocabulary**: Otomatis membangun vocab dengan token khusus (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)
- **Model Architecture**:
  - Token embeddings + sinusoidal positional encoding
  - 2–4 Transformer blocks
  - Masked self-attention
  - Feed-forward networks
  - Residual connections & Layer normalization
- **Training**: 
  - Next-token prediction dengan cross-entropy loss
  - Optimizer Adam manual dengan backprop di NumPy
  - Progress bar dan logging berwarna
  - Auto-checkpoint setiap N epoch

### 2. Inference CLI (`main.py`)

- **Interactive Chat**: CLI interaktif untuk percakapan dengan AI
- **Advanced Sampling**: 
  - Top-k sampling
  - Nucleus (top-p) sampling
  - Confidence threshold fallback
- **Colored Output**: 
  - Prompt (kuning)
  - Jawaban AI (hijau)
  - Confidence score (magenta)
- **Built-in Commands**: `help`, `config`, `exit`

## 🎯 Spesifikasi Sistem

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| **CPU** | 1 vCPU | 2+ vCPU |
| **RAM** | 2 GB | 4+ GB |
| **Storage** | 100 MB | 500 MB |
| **OS** | Termux, Linux | Any Unix-like |
| **Python** | 3.6+ | 3.8+ |
| **Dependencies** | NumPy | NumPy, tqdm, termcolor |

## ⚙️ Lingkungan yang Didukung

- 🔥 **Google Colab** dengan GPU T4 (gratis) untuk percepatan training
- ☁️ **Google Cloud Shell** (1 vCPU, 1.7 GB RAM) untuk inference ringan
- 📱 **Termux di Android** (`pkg install python numpy`)
- 🐧 **Linux/Unix** environment apapun

## 🔧 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/NeuraLite.git
cd NeuraLite
```

### 2. Install Dependencies
```bash
pip install numpy tqdm termcolor
```

### 3. Siapkan Dataset
Letakkan file dataset dengan format `*.txt` di direktori project:
```
dataset.txt
dataset2.txt
dataset_human_large.txt
```

## 🚀 Cara Penggunaan

### A. Training Model

```bash
python3 training.py \
  --data_path "*.txt" \
  --output_path slm.pkl \
  --vocab_size 5000 \
  --d_model 256 \
  --n_heads 8 \
  --n_layers 4 \
  --d_ff 1024 \
  --seq_len 128 \
  --batch_size 32 \
  --epochs 10 \
  --learning_rate 0.001 \
  --tokenizer_mode char
```

### B. Inference & Generate Teks

```bash
python3 main.py --model_path slm.pkl --confidence_threshold 0.3
```

**Contoh Penggunaan:**
```
> Halo AI, bagaimana kabarmu?
🤖 Halo! Saya baik-baik saja, terima kasih sudah bertanya. Ada yang bisa saya bantu? (confidence: 0.85)

> help
📖 Available commands:
- help: Show this help message
- config: Change generation parameters
- exit: Quit the program

> exit
👋 Sampai jumpa!
```

## 📋 Parameter Lengkap

### Training Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--data_path` | Pola file input dataset | `*.txt` |
| `--output_path` | Nama file output model | `slm.pkl` |
| `--vocab_size` | Ukuran vocabulary | `5000` |
| `--d_model` | Dimensi embedding/model | `256` |
| `--n_heads` | Jumlah attention heads | `8` |
| `--n_layers` | Jumlah Transformer blocks | `4` |
| `--d_ff` | Dimensi feed-forward | `1024` |
| `--seq_len` | Panjang sequence/context window | `128` |
| `--batch_size` | Ukuran batch training | `32` |
| `--epochs` | Jumlah epoch training | `10` |
| `--learning_rate` | Learning rate optimizer Adam | `0.001` |
| `--tokenizer_mode` | Mode tokenizer (`char`/`subword`) | `char` |

### Inference Parameters

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `--model_path` | Path file model untuk inference | `slm.pkl` |
| `--confidence_threshold` | Batas confidence fallback | `0.3` |

## 📂 Struktur Project

```
NeuraLite/
├── 🐍 training.py              # Script pelatihan model
├── 🖥️  main.py                 # Script CLI inference  
├── 📄 dataset.txt              # Contoh dataset 1
├── 📄 dataset2.txt             # Contoh dataset 2
├── 📄 dataset_human_large.txt  # Dataset besar
├── 🤖 slm.pkl                  # Hasil model training
├── 📖 README.md                # Dokumentasi project
└── 📜 LICENSE                  # MIT License
```

## 🛠️ Development

### Menjalankan Tests

```bash
# Test training dengan dataset kecil
python3 training.py --epochs 2 --batch_size 8

# Test inference
python3 main.py --model_path slm.pkl
```

### Monitoring Training

Training progress akan menampilkan:
- ✅ **[INFO]** - Informasi umum
- 🔄 **[EPOCH]** - Progress epoch?
- 📉 **[LOSS]** - Loss value per batch
- 💾 **[SAVE]** - Checkpoint saved

## 🤝 Contributing

1. **Fork** repository ini
2. **Create branch**: `git checkout -b feature-amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push branch**: `git push origin feature-amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Gunakan docstring untuk semua fungsi
- Follow PEP 8 style guide
- Tambahkan tests untuk fitur baru
- Update README jika ada perubahan API

## 📈 Roadmap

- [ ] **v1.1**: Support untuk multiple datasets format (JSON, CSV)
- [ ] **v1.2**: Web interface dengan Flask/FastAPI
- [ ] **v1.3**: Model quantization untuk ukuran lebih kecil
- [ ] **v2.0**: Support untuk fine-tuning dengan LoRA

## 🐛 Known Issues

- Training sangat lambat untuk dataset besar (>100MB)
- Memory usage tinggi untuk sequence length >256
- Belum support multi-GPU training

## 📜 License

Proyek ini didistribusikan di bawah **MIT License**. Lihat file `LICENSE` untuk detail lengkap.

## 🙏 Acknowledgments

- Terima kasih kepada komunitas NumPy untuk library yang luar biasa
- Inspired by chatGPT 4o
- Terima kasih kepada Orang orang yang telah mensupport saya.

---

## 🚨 Perhatian
- Code training ini dibuat 30% dengan AI namun dengan perhitungan yang pas
- epoch Default 10
- agar mendapatkan jawaban yang lebih natural, gunakanlah dataset traning 
<p align="center">
  <strong>Dibuat dengan ❤️ oleh Daffa</strong><br>
  <em>Siswa SMP pertama pembuat SLM di Termux</em>
</p>

<p align="center">
  <a href="#neuralite---minimalist-neural-language-model-slm-in-pure-numpy">⬆️ Back to Top</a>
</p>
