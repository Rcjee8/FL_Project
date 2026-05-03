import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

COPIES          = 6      
MIN_IMAGES      = 8      
TARGET_IMAGES   = 25     
SUFFIX          = "_aug" 
OUTPUT_EXT      = "jpg"
JPEG_QUALITY    = 90



def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def adjust_brightness(img, lo=0.5, hi=1.8):
    factor = random.uniform(lo, hi)
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img, lo=0.5, hi=1.8):
    factor = random.uniform(lo, hi)
    return ImageEnhance.Contrast(img).enhance(factor)


def adjust_saturation(img, lo=0.4, hi=1.8):
    factor = random.uniform(lo, hi)
    return ImageEnhance.Color(img).enhance(factor)


def random_rotation(img, max_deg=15):
    angle = random.uniform(-max_deg, max_deg)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def random_crop(img, crop_frac_lo=0.80, crop_frac_hi=0.95):
    w, h   = img.size
    frac   = random.uniform(crop_frac_lo, crop_frac_hi)
    cw, ch = int(w * frac), int(h * frac)
    left   = random.randint(0, w - cw)
    top    = random.randint(0, h - ch)
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


def add_gaussian_noise(img, std_lo=2, std_hi=12):
    arr  = np.array(img, dtype=np.float32)
    std  = random.uniform(std_lo, std_hi)
    arr += np.random.normal(0, std, arr.shape).astype(np.float32)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_grayscale(img, p=0.15):
    if random.random() < p:
        return img.convert("L").convert("RGB")
    return img


def slight_blur(img, p=0.25, radius_max=1.5):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.3, radius_max)))
    return img


def color_jitter(img):
    ops = [
        lambda x: adjust_brightness(x),
        lambda x: adjust_contrast(x),
        lambda x: adjust_saturation(x),
    ]
    random.shuffle(ops)
    for op in ops:
        img = op(img)
    return img



PIPELINE = [
    (horizontal_flip,   0.5),
    (random_rotation,   0.6),
    (random_crop,       0.5),
    (color_jitter,      0.8),
    (add_gaussian_noise,0.4),
    (random_grayscale,  0.15),
    (slight_blur,       0.25),
]


def augment_one(img: Image.Image) -> Image.Image:
    """Apply a random subset of augmentations to produce one variant."""
    out = img.copy()
    for fn, prob in PIPELINE:
        if random.random() < prob:
            try:
                out = fn(out)
            except Exception:
                pass 
    return out



def augment_person(person_dir: str, copies: int, target: int,
                   preview: bool = False) -> int:
    image_files = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and SUFFIX not in f 
    ]
    if not image_files:
        return 0

    existing_total = len(os.listdir(person_dir))
    if existing_total >= target:
        return 0   

    needed = target - existing_total
    written = 0

    src_cycle = image_files * ((needed // len(image_files)) + 2)
    random.shuffle(src_cycle)

    for src_name in src_cycle:
        if written >= needed:
            break

        src_path = os.path.join(person_dir, src_name)
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [warn] Cannot open {src_path}: {e}", flush=True)
            continue

        for _ in range(copies):
            if written >= needed:
                break

            aug = augment_one(img)

            stem = os.path.splitext(src_name)[0]
            out_name = f"{stem}{SUFFIX}_{written:04d}.{OUTPUT_EXT}"
            out_path = os.path.join(person_dir, out_name)

            if preview:
                aug.show()
                input("Press Enter for next preview (Ctrl-C to stop)…")
                written += 1
                continue

            try:
                aug.save(out_path, quality=JPEG_QUALITY)
                written += 1
            except Exception as e:
                print(f"  [warn] Could not save {out_path}: {e}", flush=True)

    return written



def main():
    parser = argparse.ArgumentParser(
        description="Augment an LFW-style face dataset in-place."
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Root of the dataset (one subfolder per person)."
    )
    parser.add_argument(
        "--copies", type=int, default=COPIES,
        help=f"Augmented copies to generate per source image (default {COPIES})."
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_IMAGES,
        help=f"Stop augmenting once a person has this many images (default {TARGET_IMAGES})."
    )
    parser.add_argument(
        "--min_images", type=int, default=MIN_IMAGES,
        help=f"Only augment people who have fewer than this many images (default {MIN_IMAGES})."
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show augmented images on screen instead of saving (for testing)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"Error: '{args.data_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    people = sorted([
        d for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d))
    ])

    if not people:
        print(f"No subfolders found in '{args.data_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDataset: {args.data_path}")
    print(f"People found: {len(people)}")
    print(f"Augmented copies per source: {args.copies}")
    print(f"Target images per person:    {args.target}")
    print(f"{'Preview mode' if args.preview else 'Writing to disk'}\n")

    total_before = 0
    total_after  = 0
    skipped      = 0
    augmented    = 0

    for name in people:
        person_dir = os.path.join(args.data_path, name)
        originals  = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and SUFFIX not in f
        ]
        n_orig = len(originals)
        total_before += n_orig

        if n_orig == 0:
            skipped += 1
            continue

        n_total = len(os.listdir(person_dir))   
        if n_total >= args.target:
            print(f"  {name:<35}  {n_total:>3} images — already at target, skipping")
            total_after += n_total
            skipped += 1
            continue

        print(f"  {name:<35}  {n_orig:>3} originals → augmenting …", end=" ", flush=True)
        n_written = augment_person(person_dir, args.copies, args.target, args.preview)
        n_new     = len(os.listdir(person_dir))
        print(f"+{n_written} → {n_new} total")
        total_after  += n_new
        augmented    += 1

    print(f"\n{'='*50}")
    print(f"  People processed:  {augmented}")
    print(f"  People skipped:    {skipped}")
    print(f"  Images before:     {total_before}")
    print(f"  Images after:      {total_after}")
    print(f"  New images added:  {total_after - total_before}")
    print(f"{'='*50}\n")

    if not args.preview:
        print("Done. Run phone_server.py to use the augmented dataset.")


if __name__ == "__main__":
    main()
