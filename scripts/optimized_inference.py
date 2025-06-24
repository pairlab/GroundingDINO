#!/usr/bin/env python3
"""
Optimized GroundingDINO inference script with performance improvements.
"""

import time
import torch
import torch.cuda.amp as amp
from groundingdino.util.inference import load_model, predict, load_image
import argparse
import einops

def benchmark_inference(model, image, caption, batch_size=64, num_runs=10, warmup_runs=3):
    """Benchmark inference performance with optimizations on batched images."""
    
    # Create batch of repeated images
    batched_image = einops.repeat(image, "h w c -> b h w c", b=batch_size)
    batched_caption = [caption] * batch_size
    
    print(f"Testing with batch size: {batch_size}")
    print(f"Image shape: {image.shape} -> Batched shape: {batched_image.shape}")
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = predict(model, batched_image, batched_caption, 0.35, 0.25, use_amp=True)
    
    torch.cuda.synchronize()
    
    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            boxes, logits, phrases = predict(model, batched_image, batched_caption, 0.35, 0.25, use_amp=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 5 == 0:
            print(f"Run {i+1}/{num_runs}: {times[-1]:.3f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nBenchmark Results:")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Min time: {min_time:.3f}s")
    print(f"Max time: {max_time:.3f}s")
    print(f"Throughput: {batch_size/avg_time:.2f} images/second")
    print(f"Per-image time: {avg_time/batch_size:.4f}s")
    
    return boxes, logits, phrases

def main():
    parser = argparse.ArgumentParser(description="Optimized GroundingDINO inference")
    parser.add_argument("--config", required=True, help="Model config path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--caption", default="a dog", help="Text caption")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for testing")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Load model with optimizations
    print("Loading model...")
    model = load_model(
        args.config, 
        args.checkpoint, 
        device=args.device,
        use_checkpoint=True,
        compile_model=not args.no_compile
    )
    
    # Load image
    print("Loading image...")
    image_source, image = load_image(args.image)
    
    if args.benchmark:
        # Run benchmark with batched images
        boxes, logits, phrases = benchmark_inference(
            model, image, args.caption, 
            batch_size=args.batch_size, 
            num_runs=args.num_runs
        )
    else:
        # Single batch inference
        print("Running batched inference...")
        start_time = time.time()
        
        # Create batch of repeated images
        batched_image = einops.repeat(image, "h w c -> b h w c", b=args.batch_size)
        batched_caption = [args.caption] * args.batch_size
        
        boxes, logits, phrases = predict(
            model, batched_image, batched_caption, 0.35, 0.25, 
            device=args.device, use_amp=not args.no_amp
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"Batch inference time: {end_time - start_time:.3f}s")
        print(f"Batch size: {args.batch_size}")
        print(f"Per-image time: {(end_time - start_time)/args.batch_size:.4f}s")
        print(f"Throughput: {args.batch_size/(end_time - start_time):.2f} images/second")
    
    # Print results (show first few detections)
    if boxes is not None and len(boxes) > 0:
        print(f"\nFound {len(boxes)} objects (showing first 5):")
        for i, (box, logit, phrase) in enumerate(zip(boxes[:5], logits[:5], phrases[:5])):
            print(f"  {i+1}. {phrase} (confidence: {logit:.3f})")
        if len(boxes) > 5:
            print(f"  ... and {len(boxes) - 5} more objects")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    main() 