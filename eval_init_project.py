"""
Scaffold an evaluation project under projects/<name>/.

Creates the folder layout, seeds classes.txt, and writes a config.json template
for you to fill in with your models and videos.

    python eval_init_project.py --name hph_hw \
        --classes-from D:/Dre/PDE_yolo_infra/logs/hph_hw/yolo11n2/weights/best.pt

--classes-from accepts a classes.txt, a data.yaml, or a .pt weights file.
"""

import argparse
import copy
import os
import re

from eval_common import DEFAULT_CONFIG, read_classes, sanitize, save_project

PROJECTS_ROOT = "projects"


def classes_from_data_yaml(path):
    """
    Pull the class names out of a YOLO data.yaml without importing pyyaml.

    No script in this repo depends on pyyaml -- data_merge_datasets.py even writes
    data.yaml by hand -- so we keep that constraint and parse the two shapes
    ultralytics actually emits:  names: [a, b]  and  names:\n  0: a
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        if not re.match(r"^\s*names\s*:", line):
            continue

        inline = line.split(":", 1)[1].strip()
        if inline.startswith("["):
            body = inline.strip("[]")
            return [c.strip().strip("'\"") for c in body.split(",") if c.strip()]

        indexed = {}
        for follow in lines[i + 1:]:
            if not follow.strip():
                continue
            if not follow.startswith((" ", "\t")):
                break
            m = re.match(r"^\s+'?\"?(\d+)'?\"?\s*:\s*(.+?)\s*$", follow)
            if not m:
                m2 = re.match(r"^\s+-\s*(.+?)\s*$", follow)
                if m2:
                    indexed[len(indexed)] = m2.group(1).strip("'\"")
                    continue
                break
            indexed[int(m.group(1))] = m.group(2).strip("'\"")
        if indexed:
            return [indexed[k] for k in sorted(indexed)]

    raise ValueError(f"could not find a 'names:' block in {path}")


def classes_from_weights(path):
    """Read model.names out of a .pt checkpoint."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from ultralytics import YOLO

    names = YOLO(path).names
    return [names[k] for k in sorted(names)]


def load_source_classes(path):
    lower = path.lower()
    if lower.endswith(".pt"):
        return classes_from_weights(path), "weights"
    if lower.endswith((".yaml", ".yml")):
        return classes_from_data_yaml(path), "data.yaml"
    return read_classes(path), "classes.txt"


def init_project(name, classes_source, projects_root=PROJECTS_ROOT, force=False):
    project_dir = os.path.join(projects_root, sanitize(name))
    if os.path.exists(os.path.join(project_dir, "config.json")) and not force:
        raise FileExistsError(
            f"{project_dir}/config.json already exists. Pass --force to overwrite the config "
            f"(classes.txt and gt/ are never touched)."
        )

    for sub in ("", "gt", "cache", "reports"):
        os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

    classes_path = os.path.join(project_dir, "classes.txt")
    if os.path.exists(classes_path):
        classes = read_classes(classes_path)
        print(f"📄 Keeping existing classes.txt ({len(classes)} classes)")
    else:
        if not os.path.exists(classes_source):
            raise FileNotFoundError(f"--classes-from not found: {classes_source}")
        classes, kind = load_source_classes(classes_source)
        if not classes:
            raise ValueError(f"no classes found in {classes_source}")
        with open(classes_path, "w", encoding="utf-8") as f:
            f.write("\n".join(classes) + "\n")
        print(f"📄 Wrote {len(classes)} classes from {kind}: {classes_path}")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["project"] = name
    cfg["models"] = [{"name": "REPLACE_ME", "weights": "D:/Dre/PDE_yolo_infra/logs/.../best.pt"}]
    cfg["videos"] = [{"name": "REPLACE_ME", "path": "D:/Dre/PDE_yolo_infra/video/.../clip.mp4"}]
    save_project(project_dir, cfg)

    print(f"✅ Project ready: {project_dir}")
    for line in (
        f"  1. Edit {os.path.join(project_dir, 'config.json')} -- fill in models[] and videos[].",
        "     Use forward slashes in Windows paths.",
        f"  2. python eval_run_detect.py --project {project_dir}",
        f"  3. python eval_label_timeline.py --project {project_dir} --video <video_name>",
        f"  4. python eval_report.py --project {project_dir}",
    ):
        print(line)
    return project_dir


def main():
    parser = argparse.ArgumentParser(description="Scaffold an eval project folder.")
    parser.add_argument("--name", required=True, help="Project name, e.g. hph_hw")
    parser.add_argument(
        "--classes-from",
        required=True,
        help="Source of the class list: a classes.txt, a data.yaml, or a .pt weights file",
    )
    parser.add_argument("--projects-root", default=PROJECTS_ROOT)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing config.json")
    args = parser.parse_args()

    init_project(args.name, args.classes_from, args.projects_root, args.force)


if __name__ == "__main__":
    main()
