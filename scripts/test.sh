# uv run optimized_inference.py \
#     --config groundingdino/config/GroundingDINO_SwinT_OGC.py \
#     --checkpoint weights/groundingdino_swint_ogc.pth \
#     --image /storage/home/hcoda1/1/awilcox31/vast/imitation/test_ims/demo_0/frame_0042.png \
#     --caption "left bowl . right bowl ." \
#     --benchmark \
#     --no-compile

uv run optimized_inference.py \
    --config groundingdino/config/GroundingDINO_SwinB_cfg.py \
    --checkpoint weights/groundingdino_swinb_cogcoor.pth \
    --image test.png \
    --caption "left bowl . right bowl . avocado" \
    --benchmark \
    --no-compile