from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from tqdm import trange
import einops
from pyinstrument import Profiler

model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth", compile_model=False)
IMAGE_PATH = "test.png"
TEXT_PROMPT = "left bowl. right bowl. avocado."
BOX_TRESHOLD = 0.15
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

# Count and print total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
total_params_millions = total_params / 1_000_000
print(f"Model has {total_params_millions:.1f}M parameters ({total_params:,} total)")

# image = einops.repeat(image, "h w c -> b h w c", b=64)
# caption = [TEXT_PROMPT] * 64
caption = TEXT_PROMPT

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=caption,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# Print results (show first few detections)
if boxes is not None and len(boxes) > 0:
    print(f"\nFound {len(boxes)} objects (showing first 5):")
    for i, (box, logit, phrase) in enumerate(zip(boxes[:5], logits[:5], phrases[:5])):
        print(f"  {i+1}. {phrase} (confidence: {logit:.3f})")
    if len(boxes) > 5:
        print(f"  ... and {len(boxes) - 5} more objects")
else:
    print("No objects detected.")

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)