Ý tưởng này **rất nên làm**, đặc biệt để paper/ablation có cơ sở. Bạn đang có model mạnh final, nhưng để chứng minh **Structural Feature / TopoAdapter thật sự giúp gì**, nên train thêm các option ablation.

Mình đề xuất chia thành 2 nhóm:

---

## Nhóm A — baseline không structural adapter

### A0. ViT pretrained + Qwen + LoRA, không OT, không structural loss

Mục tiêu: baseline chính để trả lời câu hỏi:

> Nếu chỉ dùng image encoder pretrained + Qwen thì score bao nhiêu?

Config nên là:

```bash
--vision-pretrained true
--freeze-vision-backbone true
--use-lora true
--use-ot false
--use-ot-fusion false
--use-topological-loss false
--use-prior-align-loss false
--use-global-topo-loss false
--use-patch-topo-loss false
```

Model này là baseline “clean”.

---

## Nhóm B — bật từng structural feature

### B1. Prior mask only

Mục tiêu: xem lesion prior có giúp không.

```bash
--use-ot true
--use-ot-fusion true
--use-prior-as-ot-target true
--use-prior-align-loss true
--use-global-topo-loss false
--use-patch-topo-loss false
```

### B2. Global topology only

Mục tiêu: xem global topo vector giúp không.

```bash
--use-ot false hoặc true tùy bạn muốn tách sạch
--use-global-topo-loss true
--use-patch-topo-loss false
```

### B3. Patch/TDA topology only

Mục tiêu: xem TDA patch feature giúp không.

```bash
--use-patch-topo-loss true
--use-global-topo-loss false
--use-prior-align-loss false
```

### B4. Full structural features

Đây là model CATA hiện tại:

```bash
--topo-mode all
--use-prior-as-ot-target true
--use-prior-align-loss true
--use-global-topo-loss true
--use-patch-topo-loss true
```

---

## Nhóm C — TopoAdapter ablation

### C1. Chỉ TDA Adapter + ViT pretrained + Qwen

Đây là option bạn nói rất đúng. Nó trả lời câu hỏi:

> Adapter topology tự nó có đóng góp không, khi không dùng full prior/global?

Config:

```bash
--topo-mode tda_only
--adapter-last-n-layers 8
--bottleneck-dim 32
--vision-pretrained true
--freeze-vision-backbone true
```

Trong code train hiện tại [train_qwen3b_curriculum_topo_adapter.py](file:///C:/Users/nguye/Project/medico_vqa_2026/scripts/train_qwen3b_curriculum_topo_adapter.py), `tda_only` sẽ:

```python
prior_mask = torch.zeros_like(prior_mask)
global_features = torch.zeros_like(global_features)
```

và adapter condition chỉ dùng:

```python
mean/std/max của topo_features
```

tức là đúng “chỉ TDA Adapter”.

---

# Bảng ablation nên có trong paper

Bạn có thể báo cáo kiểu này:

| ID | ViT pretrained | Qwen LoRA | Prior OT | Global topo | Patch TDA | TopoAdapter | Test fine-tune | METEOR | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ... | ... |
| B1 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ... | ... |
| B2 | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ... | ... |
| B3 | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ... | ... |
| C1 | ✅ | ✅ | ❌ | ❌ | ✅-cond | ✅ | ❌ | ... | ... |
| B4/CATA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ... | ... |
| Final | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ... | ... |

---

# Nhưng nên tránh train quá nhiều full runs

Vì mỗi run khá tốn GPU, mình khuyên:

## Phase 1 — train nhanh trên 30k hoặc 10k

Chạy nhanh để lọc:

- A0 baseline
- B1 prior only
- B3 patch TDA only
- C1 TDA adapter only
- Full CATA

Ví dụ mỗi option:

```bash
--max-samples 30000
--epochs 1 hoặc 2
```

## Phase 2 — chỉ chọn top 2 train full

Sau Phase 1 chọn top 2 hoặc top 3:

- Baseline tốt nhất
- TDA Adapter only
- Full CATA

rồi train full train data.

## Phase 3 — chỉ model tốt nhất train thêm full test

Chỉ dùng test fine-tune cho final submission, không cần làm với tất cả.

---

# Cấu hình mình khuyên ưu tiên

Nếu thời gian/GPU hạn chế, ưu tiên 4 model:

## 1. Baseline

```text
ViT pretrained + Qwen LoRA
```

## 2. A3 prior OT fusion

```text
ViT pretrained + Qwen LoRA + prior OT fusion
```

## 3. TDA Adapter only

```text
ViT pretrained + Qwen LoRA + TDA Adapter
```

## 4. Full CATA

```text
ViT pretrained + Qwen LoRA + Prior + Global + Patch TDA + TopoAdapter
```

Chỉ 4 model này là đủ làm paper đẹp rồi.

---

# Kết luận

Bạn nên làm. Đây là hướng đúng để chứng minh đóng góp:

1. **Baseline:** ViT pretrained + Qwen.
2. **Structural individual:** prior/global/TDA từng cái.
3. **TDA Adapter only:** ablation quan trọng nhất.
4. **Full CATA:** model chính.
5. **Final test-finetune:** chỉ cho leaderboard, report riêng.

Nếu muốn, bước tiếp theo mình có thể viết cho bạn một script kiểu:

```text
scripts/run_cata_ablation_training.py
```

để tự chạy tuần tự các option này, lưu checkpoint/eval/report thành bảng.