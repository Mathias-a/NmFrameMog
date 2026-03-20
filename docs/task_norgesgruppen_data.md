# NorgesGruppen Data

This package implements the required offline runner contract:

```bash
python run.py --input /data/images --output /output/predictions.json
```

## What this first pass does

- Scans the input directory for common image file formats.
- Reads each image size with Pillow.
- Emits one deterministic COCO-style detection per image.
- Derives `image_id` from the last integer in the filename stem, with a sequential fallback when no integer is present.

The output file is always a JSON array of objects in the required shape:

```json
[
  {
    "image_id": 12,
    "category_id": 1,
    "bbox": [5, 2, 10, 5],
    "score": 0.5
  }
]
```

## Notes

This is delivery infrastructure plus a deterministic baseline, not a trained detector. It is intentionally honest about the missing model specification while still producing valid submission output.
