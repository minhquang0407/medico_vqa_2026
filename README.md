```text
medico_vqa_2026/
├── data/                         # Dữ liệu cục bộ (Bị chặn bởi .gitignore)
│   ├── raw/                      # Kvasir-VQA-x1 gốc (Images & JSONL)
│   ├── augmented/                # Dữ liệu sinh ra từ Albumentations [Role 1]
│   └── topo_cache/               # Cache ma trận đặc trưng TDA (.npy) [Role 1]
│
├── visuals/                      # Kết quả trực quan hóa (Heatmaps, Plot) [Role 2]
│
├── src/                          # MÃ NGUỒN CHÍNH
│   ├── __init__.py
│   ├── data_pipeline/            # [ROLE 1] Hậu cần dữ liệu
│   │   ├── augmentation.py       # Xử lý Albumentations Offline
│   │   └── dataset.py            # Custom Pytorch Dataloader
│   │
│   ├── topology/                 # [ROLE 1] Toán học Hình học
│   │   └── tda_extractor.py      # Tính Persistent Homology & Bottleneck Distance
│   │
│   ├── alignment/                # [ROLE 1] Toán học Vận tải
│   │   └── sinkhorn_ot.py        # Sinkhorn-Knopp & Graph Optimal Transport (GOT)
│   │
│   ├── models/                   # [ROLE 2] Kiến trúc Mạng Nơ-ron
│   │   ├── llm_decoder.py        # Llama-3 Q-LoRA Adapter
│   │   └── fusion.py             # Cơ chế gộp Modality (Cross-Attention/Gating)
│   │
│   ├── inference/                # [ROLE 2] Suy luận & Giải thích
│   │   ├── bayesian_gate.py      # Ước lượng độ tin cậy (Posterior Confidence)
│   │   └── visualizer.py         # Trực quan hóa bản đồ nhiệt (Grad-CAM/Attention)
│   │
│   └── training/                 # [ROLE 2] Huấn luyện
│       └── train_loop.py         # Curriculum Learning & Backpropagation
│
├── notebooks/                    # Thử nghiệm thuật toán & POC (.ipynb)
├── scripts/                      # Bash scripts tự động hóa (.sh)
│   ├── run_augmentation.sh
│   └── run_training.sh
│
├── .gitignore                    # Quản lý loại trừ file (data/, visuals/, __pycache__/)
├── requirements.txt              # Danh sách thư viện (Giotto-TDA, POT, Peft...)
└── README.md                     # Tài liệu hướng dẫn dự án
```
