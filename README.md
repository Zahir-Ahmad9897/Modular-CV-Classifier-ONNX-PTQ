# Modular Image Classification — Training + ONNX/PTQ Deployment

Reusable PyTorch pipeline for image classification: swappable backbones, swappable datasets, config-driven training, and a deployment path through ONNX export and post-training quantization.

## Features

- **Backbones**: custom CNN · ConvNeXt-Tiny · ViT-B/16 · Faster R-CNN backbone (repurposed for classification)
- **Datasets**: CIFAR-10 (extensible to CIFAR-100, STL-10)
- **Training toggles**: AMP · cosine LR scheduler · gradient clipping · weight decay · backbone freezing · seeding
- **Evaluation**: per-class + macro/weighted metrics, confusion matrix plots
- **Deployment**: ONNX export, dynamic INT8 PTQ, static INT8 PTQ (calibrated), FP32 vs. INT8 comparison report

## Structure

```
datasets.py           dataset factory
models.py              model factory
train.py                training loop + toggles + checkpointing
eval.py                 evaluate a .pt checkpoint or .onnx model
export_onnx.py     PyTorch -> ONNX export
quantize.py           dynamic / static INT8 quantization
utils/metrics.py    metrics + confusion matrix
part_1_results/     per-experiment metrics + confusion matrices
part_2_results/     FP32 vs INT8 comparison report
```

## Setup

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

## Usage

```bash
# train
python train.py --arch custom_cnn --dataset cifar10 --data_dir ./data \
  --epochs 10 --use_amp 1 --use_scheduler 1 --use_grad_clip 1 \
  --use_weight_decay 1 --freeze_backbone 0 --seed 42

# evaluate (checkpoint or ONNX)
python eval.py --ckpt best_model.pt --arch custom_cnn --dataset cifar10 --data_dir ./data
python eval.py --ckpt model.onnx --dataset cifar10 --data_dir ./data

# export
python export_onnx.py --ckpt best_model.pt --arch custom_cnn --output model.onnx

# quantize
python quantize.py --mode dynamic --input model.onnx --output model_dynamic_int8.onnx
python quantize.py --mode static  --input model.onnx --output model_static_int8.onnx --data_dir ./data
```

`--arch`: `custom_cnn` · `convnext_tiny` · `vit_b_16` · `fasterrcnn_backbone`

## Notes

- Faster R-CNN backbone: detection heads discarded; one FPN feature level is pooled to a fixed vector and fed to a new linear head.
- Input resolution is architecture-dependent (32x32 for `custom_cnn`, 224x224 for the pretrained backbones) via `image_size` in the dataset factory.
- Static quantization calibrates on the validation split.

## License

MIT
