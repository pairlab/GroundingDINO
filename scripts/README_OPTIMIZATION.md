# GroundingDINO Optimized Inference

This document provides examples and usage instructions for the optimized GroundingDINO inference with performance improvements.

## Quick Start

### Basic Inference
```bash
# Single image inference with optimizations enabled
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image path/to/your/image.jpg \
    --caption "a dog and a cat"
```

### Performance Benchmarking
```bash
# Run comprehensive benchmark (10 iterations with warmup)
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image path/to/your/image.jpg \
    --caption "bowl cup spoon" \
    --benchmark
```

## Advanced Usage Examples

### 1. Object Detection in Kitchen Scene
```bash
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image kitchen_scene.jpg \
    --caption "knife fork spoon plate bowl cup"
```

### 2. Person and Object Detection
```bash
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image street_scene.jpg \
    --caption "person car bicycle traffic light"
```

### 3. Animal Detection
```bash
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image zoo.jpg \
    --caption "elephant lion tiger giraffe"
```

### 4. Furniture Detection
```bash
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image living_room.jpg \
    --caption "chair table sofa lamp television"
```

## Performance Testing

### Compare Optimizations
```bash
# Test with all optimizations (default)
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image test_image.jpg \
    --caption "test objects" \
    --benchmark

# Test without torch.compile
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image test_image.jpg \
    --caption "test objects" \
    --benchmark \
    --no-compile

# Test without mixed precision
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image test_image.jpg \
    --caption "test objects" \
    --benchmark \
    --no-amp

# Test without any optimizations
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image test_image.jpg \
    --caption "test objects" \
    --benchmark \
    --no-compile \
    --no-amp
```

### Batch Processing Example
```bash
# Process multiple images with the same caption
for img in images/*.jpg; do
    python optimized_inference.py \
        --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
        --checkpoint weights/groundingdino_swinb_cogcoor.pth \
        --image "$img" \
        --caption "person chair table"
done
```

## Using the Original Test Script with Optimizations

You can also use the original test script with the new optimizations:

```python
# Modified test.py with optimizations
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from tqdm import trange
import einops
from pyinstrument import Profiler

# Load model with optimizations
model = load_model(
    "groundingdino/config/GroundingDINO_SwinB_cfg.py", 
    "weights/groundingdino_swinb_cogcoor.pth",
    use_checkpoint=True,  # Enable checkpointing
    compile_model=True    # Enable torch.compile
)

IMAGE_PATH = "path/to/your/image.jpg"
TEXT_PROMPT = "left bowl . right bowl ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
total_params_millions = total_params / 1_000_000
print(f"Model has {total_params_millions:.1f}M parameters ({total_params:,} total)")

# Batch processing
image = einops.repeat(image, "h w c -> b h w c", b=64)
caption = [TEXT_PROMPT] * 64

# Profiling with optimizations
profiler = Profiler()
profiler.start()
for _ in trange(10):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=caption,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        use_amp=True  # Enable mixed precision
    )
profiler.stop()

print(profiler.output_text(show_all=True, color=True))
```

## Expected Performance Improvements

- **Mixed Precision (AMP)**: 15-30% speedup
- **Torch Compile**: 10-20% speedup
- **Gradient Checkpointing**: Better memory efficiency

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Try reducing batch size or using CPU
2. **Torch compile fails**: Use `--no-compile` flag
3. **Mixed precision errors**: Use `--no-amp` flag

### Memory Optimization
```bash
# For limited GPU memory
export CUDA_VISIBLE_DEVICES=0
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image large_image.jpg \
    --caption "objects" \
    --no-compile  # Disable compile if memory is limited
```

### CPU Fallback
```bash
# Use CPU if GPU is not available
python optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image image.jpg \
    --caption "objects" \
    --device cpu
``` 