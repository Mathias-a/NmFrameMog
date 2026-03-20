## NOTES

Structure should be something along this:
- One main branch that holds the general infrastructure of training for object detection. That should be infrastructure that is shared across 
all models. Dont know yet what this can be.
- One branch per model, which exist in its own worktree. The motivation for this is to be able to run multiple models at once. I have very generous resources at gcp.
In gcp, i can create one VM per model, and run containerized training jobs in there via the gcloud CLI from my laptop terminal. 
-  I am imagining that each model gets its own folder under src/object_detection (e.g. src/object_detection/yolo)

## Requirements

- The submission should be a .zip file containing run.py at root, as well as optional model weights file and utils.py file. 
- The run.py file should be ran like python run.py --input /data/images --output /output/predictions.json
- The input parameter gives a path to images in the structure img_XXXXX.jpg (e.g., img_00042.jpg).
- The output parameter gives an output path with a .json of predictions, in the form of:

| Field | Type | Description |
|---|---|---|
| `image_id` | `int` | Numeric ID from filename (e.g., `img_00042.jpg` → `42`) |
| `category_id` | `int` | Product category ID (0-355). See categories list in `annotations.json` |
| `bbox` | `[x, y, w, h]` | Bounding box in COCO format |
| `score` | `float` | Confidence score (0-1) |

- I want to have a structure where i use .pt file to store the model weights, and then load it in run.py. Requirement is that the model should only use standard pytorch ops
- Cannot pip install at runtime
- The sandbox has a GPU (NVIDIA L4 with CUDA 12.4), so GPU-trained weights run natively — no map_location="cpu" needed. Your code should auto-detect with torch.cuda.is_available().



