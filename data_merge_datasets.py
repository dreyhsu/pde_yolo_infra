import os
import shutil

# Datasets to merge (in order). Filenames are already prefixed per-dataset,
# so images/labels can be copied flat without renaming.
SOURCE_DIRS = [
    os.path.join("transfer", "hph_packing_1"),
    os.path.join("transfer", "hph_packing_2"),
    os.path.join("transfer", "hph_packing_3"),
]

# Merged output folder.
OUTPUT_DIR = os.path.join("transfer", "hph_packing_merged")

# Dataset whose classes.txt defines the canonical class order for the merged set.
REFERENCE_DIR = os.path.join("transfer", "hph_packing_2")

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def normalize(name):
    """Normalize a class name so e.g. 'charging-cable' matches 'chargingcable'."""
    return name.strip().lower().replace('-', '').replace('_', '').replace(' ', '')


def read_classes(dataset_dir):
    """Read classes.txt into an ordered list of class names."""
    path = os.path.join(dataset_dir, "classes.txt")
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def build_remap(source_classes, name_to_target):
    """Map each source class id -> canonical target id via normalized name."""
    remap = {}
    for src_id, name in enumerate(source_classes):
        key = normalize(name)
        if key not in name_to_target:
            raise ValueError(
                f"Class '{name}' not found in canonical classes. "
                f"Update {REFERENCE_DIR}/classes.txt or check the source."
            )
        remap[src_id] = name_to_target[key]
    return remap


def remap_label_file(src_path, dst_path, remap):
    """Rewrite the leading class id of each line, preserving bbox coords."""
    with open(src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        parts[0] = str(remap[src_id])
        out_lines.append(" ".join(parts))

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")


def merge_datasets():
    # Canonical class list + normalized-name lookup.
    canonical = read_classes(REFERENCE_DIR)
    name_to_target = {normalize(n): i for i, n in enumerate(canonical)}
    print(f"Canonical classes ({len(canonical)}): {canonical}")

    out_images = os.path.join(OUTPUT_DIR, "images")
    out_labels = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    total_images = 0
    total_labels = 0

    for src_dir in SOURCE_DIRS:
        src_classes = read_classes(src_dir)
        remap = build_remap(src_classes, name_to_target)
        is_identity = all(k == v for k, v in remap.items())

        images_dir = os.path.join(src_dir, "images")
        labels_dir = os.path.join(src_dir, "labels")

        img_count = 0
        lbl_count = 0

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(IMAGE_EXTS):
                continue

            # Copy image (keep name; already unique across datasets).
            shutil.copy2(
                os.path.join(images_dir, img_name),
                os.path.join(out_images, img_name),
            )
            img_count += 1

            # Remap matching label if it exists; images without labels are
            # kept as background/negative examples.
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_name)
            if os.path.exists(src_label):
                remap_label_file(
                    src_label,
                    os.path.join(out_labels, label_name),
                    remap,
                )
                lbl_count += 1
            else:
                print(f"  Note: no label for {img_name} (kept as background)")

        total_images += img_count
        total_labels += lbl_count
        print(
            f"{src_dir}: {img_count} images, {lbl_count} labels "
            f"({'identity' if is_identity else 'remapped'})"
        )

    # Write merged classes.txt and data.yaml.
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(canonical) + "\n")

    with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w', encoding='utf-8') as f:
        f.write("path: " + os.path.abspath(OUTPUT_DIR) + "\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        for i, name in enumerate(canonical):
            f.write(f"  '{i}': {name}\n")

    print(
        f"\nDone. Merged {total_images} images and {total_labels} labels "
        f"into {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    merge_datasets()
