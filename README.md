# ObjectSeqEmotion

> Predict **positive / neutral / negative** emotions from the **order in which a person looks at objects**.

## Quick Start

```bash
# 1. Install dependencies (using Tsinghua mirror for speed)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. Object detection → per‑frame CSVs
python scripts/extract_objects.py \
  --frames_root videos/frames \
  --save_dir    data/objects_csv

# 3. Build one‑line object sequences
python scripts/build_seq_csv.py \
  --csv_root data/objects_csv \
  --save_dir data/seq_csv

# 4. Label emotions (interactive CLI)
python scripts/label_emotion_csv.py \
  --seq_root  data/seq_csv \
  --save_path data/emotion_labels.csv

# 5. Train the model
python scripts/train_CSV.py \
  --seq_root  data/seq_csv \
  --label_csv data/emotion_labels.csv

# 6. Evaluate / infer
python scripts/val_CSV.py \
  --checkpoint runs/exp/best_model.pth \
  --seq_root  data/seq_csv
```

## Project Layout

```
ObjectSeqEmotion/
├─ scripts/      # end‑to‑end pipeline scripts
├─ models/       # ObjectSeqEmotionModel (CNN+GRU by default)
├─ datasets/     # put your data here
└─ requirements.txt
```

## Customisation

* **Model**: edit `models/network.py` to swap in a Transformer, TCN, etc.
* **Detector**: pass your own weight file to `extract_objects.py --weights yolov8*.pt`.

## Notes

* Each row in the sequence CSV equals one video clip.
* Emotion labels file format: `video_id,emotion` where emotion ∈ {0: negative, 1: neutral, 2: positive}.

Need more details? Let me know and I can expand specific sections.
