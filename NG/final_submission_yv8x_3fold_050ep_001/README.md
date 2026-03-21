# Final Submission: `yv8x_3fold_050ep_001`

This folder contains the local submission-builder and inference code for the selected final model pair:

- detector: `yv8x_3fold_050ep_001` fold `1`
- recognizer: `yv8x_3fold_050ep_001` fold `2`

The actual weight files are expected locally under:

```text
NG/yolov8x_product_pipeline/weights/yv8x_3fold_050ep_001/
  detector_fold1_best.pt
  recognizer_fold2_best.pt
```

They are intentionally ignored by git.

## Build the zip-ready submission folder locally

```bash
python NG/final_submission_yv8x_3fold_050ep_001/build_submission.py
```

That creates:

```text
NG/final_submission_yv8x_3fold_050ep_001/dist/
  yv8x_3fold_050ep_001_submission/
    run.py
    recognizer_model.py
    detector.pt
    recognizer.pt
```

and also:

```text
NG/final_submission_yv8x_3fold_050ep_001/dist/yv8x_3fold_050ep_001_submission.zip
```

## Local sanity test

Create a 130-image subset and run the staged submission package:

```bash
python NG/final_submission_yv8x_3fold_050ep_001/local_sanity_check.py \
  --train-image-dir /Users/matiasfernandezjr/Data/NM/train/images \
  --subset-size 130
```

The script:

- builds the submission staging folder
- creates a local 130-image input subset
- runs `run.py --input ... --output ...`
- validates output JSON shape
- prints runtime and prediction count

## Submission constraints addressed

- `run.py` lives at zip root
- only `.py`, `.json`, `.pt` files are staged
- inference only, no training code
- uses only allowed imports in submission code
- output follows the competition JSON format
