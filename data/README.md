# Data setup

This project uses COCO-format instance segmentation annotations. I trained on OCID (Object Clutter Indoor Dataset) converted to COCO format, but you can plug in any COCO-format segmentation dataset.

## Option 1: OCID dataset (what I used)

1. Download OCID from https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
2. Extract into `data/raw/OCID/`
3. Run the conversion script:

```bash
python scripts/convert_ocid.py --ocid-root data/raw/OCID --output-dir data/ --val-split 0.15 --test-split 0.15
```

This creates the train/val/test split with COCO annotations.

## Option 2: your own data

Just make sure the directory looks like this:

```
data/
  train/
    images/        # all training images
    annotations.json   # COCO-format with segmentation masks
  val/
    images/
    annotations.json
  test/
    images/
    annotations.json
```

The annotations.json should follow standard COCO format — `images`, `annotations` (with `segmentation` polygons and `bbox`), and `categories` fields.

## Notes

- Images get resized during training (see config), so resolution doesn't matter much
- I filtered out objects smaller than 32x32 px during conversion — too small to mask reliably
- The OCID scenes are top-down views of tabletop clutter which is close enough to warehouse bin views for our purposes
