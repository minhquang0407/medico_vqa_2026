medico_vqa_2026/
│
├── data/                       # Chứa dữ liệu (Tuyệt đối không push lên Git)
│   ├── raw/                    # Kvasir-VQA-x1 gốc (ảnh và JSONL)
│   ├── augmented/              # Dữ liệu sinh ra từ Albumentations (Role 1)
│   └── topo_cache/             # Các file ma trận .npy của TDA (Role 1)
│
├── visuals/                    # Thư mục lưu ảnh Heatmap xuất ra (Role 2)
│
├── src/                        # THƯ MỤC MÃ NGUỒN CHÍNH
│   ├── __init__.py
│   │
│   ├── data_pipeline/          # [ROLE 1] Hậu cần dữ liệu
│   │   ├── __init__.py
│   │   ├── augmentation.py     # Lệnh chạy Albumentations Offline
│   │   └── dataset.py          # Class TopologicalVQADataset (Dataloader)
│   │
│   ├── topology/               # [ROLE 1] Toán học Hình học
│   │   ├── __init__.py
│   │   └── tda_extractor.py    # Tính Persistent Homology & lưu Cache
│   │
│   ├── alignment/              # [ROLE 1] Toán học Vận tải
│   │   ├── __init__.py
│   │   └── sinkhorn_ot.py      # Thuật toán Sinkhorn-Knopp & Loss GOT
│   │
│   ├── models/                 # [ROLE 2] Kiến trúc Mạng Nơ-ron
│   │   ├── __init__.py
│   │   ├── llm_decoder.py      # Cấu hình Llama-3 Q-LoRA
│   │   └── fusion.py           # Hàm trộn modality (fuse_modalities)
│   │
│   ├── inference/              # [ROLE 2] Suy luận & Giải thích
│   │   ├── __init__.py
│   │   ├── bayesian_gate.py    # Tính Posterior Confidence
│   │   └── visualizer.py       # Vẽ Heatmap bằng Matplotlib
│   │
│   └── training/               # [ROLE 2] Huấn luyện
│       ├── __init__.py
│       └── train_loop.py       # Vòng lặp Curriculum Learning & Backprop
│
├── notebooks/                  # Các file Jupyter để test toán học (.ipynb)
├── scripts/                    # Các file bash (.sh) để chạy tự động
│   ├── run_augmentation.sh
│   └── run_training.sh
│
├── .gitignore                  # Chặn upload data/, visuals/, __pycache__/
├── requirements.txt            # Danh sách thư viện (PyTorch, Giotto-TDA, Albumentations...)
└── README.md                   # Tài liệu Kick-off dự án
