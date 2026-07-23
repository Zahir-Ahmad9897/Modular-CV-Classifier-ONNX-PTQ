# Modular Image Classification — Training + ONNX/PTQ Deployment

A PyTorch pipeline for image classification: swappable backbones, swappable datasets, config-driven training, and a deployment path through ONNX export and post-training quantization.

Implemented and run as a single Google Colab notebook (`notebook.ipynb`), organized into clearly labeled sections corresponding to each pipeline stage.

## Features

- **Backbones**: custom CNN · ConvNeXt-Tiny · ViT-B/16 · Faster R-CNN backbone (repurposed for classification)
- **Datasets**: CIFAR-10 (extensible to CIFAR-100, STL-10)
- **Training toggles**: AMP · cosine LR scheduler · gradient clipping · weight decay · backbone freezing · seeding
- **Evaluation**: per-class + macro/weighted metrics, confusion matrix plots
- **Deployment**: ONNX export, dynamic INT8 PTQ, static INT8 PTQ (calibrated), FP32 vs. INT8 comparison report

## Repository structure

```
notebook.ipynb        full pipeline: data, models, metrics, training, ONNX, quantization
part_1_results/     per-experiment metrics + confusion matrices
part_2_results/     FP32 vs INT8 comparison report
requirements.txt
README.md
```

## Notebook sections

1. **Setup** — Drive mount, imports
2. **`datasets.py` (cell)** — `make_downloader(name, data_dir, batch_size, image_size)`: dataset factory, CIFAR-10
3. **`models.py` (cell)** — `make_model(arch, num_classes, pretrained)`: model factory for all four backbones
4. **`utils/metrics.py` (cell)** — `compute_metrics`, `save_confusion_matrix`
5. **`train.py` (cell)** — `setup_training`, `train_epochs`, `evaluate`, checkpointing, config-driven toggles
6. **Part 1 results** — trains one experiment, writes `part_1_results/summary.txt` and confusion matrix
7. **`export_onnx.py` (cell)** — PyTorch → ONNX export with a PyTorch-vs-ONNX numerical sanity check
8. **`quantize.py` (cell)** — dynamic and static (calibrated) INT8 quantization
9. **Part 2 results** — evaluates PyTorch FP32, ONNX dynamic INT8, and ONNX static INT8 on the same model/dataset, writes `part_2_results/summary.txt`

## Setup

```bash
git clone https://github.com/Zahir-Ahmad9897/Modular-CV-Classifier-ONNX-PTQ.git
cd Modular-CV-Classifier-ONNX-PTQ
pip install -r requirements.txt
```

Open `notebook.ipynb` in Google Colab or Jupyter. Update `data_dir` in the setup cell to point to your local CIFAR-10 directory, then run cells top to bottom.

## Config toggles (train section)

```python
config = {
    "lr": 0.001,
    "use_weight_decay": True,
    "use_scheduler": True,
    "use_amp": True,
    "use_grad_clip": True,
    "freeze_backbone": False,
    "seed": 42,
    "epochs": 10,
}
```

`arch` options: `custom_cnn` · `convnext_tiny` · `vit_b_16` · `fasterrcnn_backbone`

## Results

- `part_1_results/summary.txt` — per-class and global metrics per experiment
- `part_1_results/cn_{arch}_{dataset}.png` — confusion matrix per experiment
- `part_2_results/summary.txt` — accuracy, macro F1, weighted F1 for PyTorch FP32 vs. ONNX dynamic INT8 vs. ONNX static INT8

## Notes

- Faster R-CNN backbone: detection heads discarded; one FPN feature level is pooled to a fixed vector and fed to a new linear head.
- Input resolution is architecture-dependent (32x32 for `custom_cnn`, 224x224 for the pretrained backbones), set via `image_size` in the dataset factory.
- Static quantization calibrates on the validation split.
- Training was run on limited compute (CPU); reported metrics reflect that constraint.

## License

MIT
