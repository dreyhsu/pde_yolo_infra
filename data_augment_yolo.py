import os
import cv2
import argparse
import random
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
import albumentations as A

def load_yolo_annotations(label_path):
    """載入 YOLO 格式的標籤檔"""
    annotations = []
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                bboxes.append(bbox)
                class_labels.append(class_id)
    return bboxes, class_labels, 'yolo'

def save_yolo_annotations(label_path, bboxes, class_labels, **kwargs):
    """儲存 YOLO 格式的標籤檔"""
    with open(label_path, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{int(label)} {' '.join(map(str, bbox))}\n")

def load_pascal_voc_annotations(xml_path):
    """載入 PASCAL VOC (.xml) 格式的標籤檔"""
    bboxes = []
    class_labels = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
        class_labels.append(name)
    return bboxes, class_labels, 'pascal_voc'

def save_pascal_voc_annotations(output_path, bboxes, class_labels, filename, width, height, depth=3):
    """儲存 PASCAL VOC (.xml) 格式的標籤檔"""
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'augmented'
    ET.SubElement(annotation, 'filename').text = filename
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    for bbox, label in zip(bboxes, class_labels):
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = str(label)
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def load_cvat_annotations(xml_path):
    """載入 CVAT for Images (.xml) 格式的標籤檔"""
    bboxes = []
    class_labels = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # CVAT XML 的結構是 <annotations><image>...</image></annotations>
    # 我們假設一個 XML 對應一張圖片
    for image_tag in root.findall('image'):
        for box in image_tag.findall('box'):
            label = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            bboxes.append([xtl, ytl, xbr, ybr])
            class_labels.append(label)
    return bboxes, class_labels, 'pascal_voc' # CVAT box 格式與 pascal_voc 相同

def save_cvat_annotations(output_path, bboxes, class_labels, filename, width, height):
    """儲存 CVAT for Images (.xml) 格式的標籤檔"""
    annotations = ET.Element('annotations')
    version = ET.SubElement(annotations, 'version')
    version.text = '1.1'
    image = ET.SubElement(annotations, 'image', id='0', name=filename, width=str(width), height=str(height))

    for i, (bbox, label) in enumerate(zip(bboxes, class_labels)):
        ET.SubElement(image, 'box', label=str(label), occluded='0', source='manual',
                      xtl=str(bbox[0]), ytl=str(bbox[1]),
                      xbr=str(bbox[2]), ybr=str(bbox[3]), z_order='0')

    tree = ET.ElementTree(annotations)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def detect_and_load_annotations(image_path, labels_dir):
    """自動偵測並載入對應的標籤檔"""
    base_name = Path(image_path).stem
    
    # 優先尋找 XML，再找 TXT
    xml_path = Path(labels_dir) / f"{base_name}.xml"
    if xml_path.exists():
        try:
            # 嘗試解析為 CVAT 格式
            tree = ET.parse(xml_path)
            if tree.getroot().tag == 'annotations':
                bboxes, labels, fmt = load_cvat_annotations(xml_path)
                return bboxes, labels, fmt, save_cvat_annotations, '.xml'
            # 否則，解析為 PASCAL VOC 格式
            else:
                bboxes, labels, fmt = load_pascal_voc_annotations(xml_path)
                return bboxes, labels, fmt, save_pascal_voc_annotations, '.xml'
        except ET.ParseError:
            print(f"警告：XML檔案 {xml_path} 解析失敗，跳過。")
            return None, None, None, None, None

    txt_path = Path(labels_dir) / f"{base_name}.txt"
    if txt_path.exists():
        bboxes, labels, fmt = load_yolo_annotations(txt_path)
        return bboxes, labels, fmt, save_yolo_annotations, '.txt'

    return None, None, None, None, None

def main(args):
    """主程式"""
    # 建立輸出資料夾
    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_labels, exist_ok=True)

    # --- 根據參數動態建立擴增流程 ---
    def create_transforms(rotate_limit=None, rotate_step=None):
        # 基本擴增流程 (已調降強度以避免類別混淆)
        base_transforms = [
            A.HorizontalFlip(p=0.5),
            # 降低透視變換強度，避免小零件變形過度
            A.Perspective(scale=(0.01, 0.04), p=0.2),
            # 降低亮度和對比度變動，保留零件邊緣與陰影特徵
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            # 移除 HueSaturationValue 以避免零件顏色與背景混淆
        ]

        if rotate_step is not None:
            # 使用者指定了固定角度步長旋轉
            print(f"啟用固定角度步長旋轉，步長: {rotate_step} 度")
            # 這裡只回傳基本擴增，旋轉將在主迴圈中處理
            rotation_transforms = []
        elif rotate_limit is not None:
            # 使用者指定了隨機旋轉角度
            print(f"啟用隨機角度旋轉，範圍: +/- {rotate_limit} 度")
            rotation_transforms = [
                A.Rotate(limit=rotate_limit, p=0.7, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
            ]
        else:
            # 預設使用 90/180/270 度旋轉
            print("啟用 90/180/270 度固定角度旋轉")
            rotation_transforms = [
                A.RandomRotate90(p=0.75), # 以 75% 的機率進行 90/180/270 度旋轉
            ]

        # 組合擴增流程
        all_transforms = rotation_transforms + base_transforms
        transform_yolo = A.Compose(all_transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))
        transform_voc = A.Compose(all_transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.2))
        return transform_yolo, transform_voc

    transform_yolo, transform_voc = create_transforms(args.rotate_limit, args.rotate_step)

    image_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(image_files)} 張影像，開始進行擴增...")

    for filename in tqdm(image_files):
        # 讀取影像
        image_path = str(Path(args.images_dir) / filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取影像 {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # 自動偵測並載入標籤
        bboxes, class_labels, ann_format, save_func, ann_ext = detect_and_load_annotations(image_path, args.labels_dir)

        if bboxes is None:
            print(f"警告：找不到影像 {filename} 對應的標籤檔，跳過。")
            continue

        # 根據格式選擇擴增流程
        transform = transform_voc if ann_format == 'pascal_voc' else transform_yolo

        # 決定擴增的迭代次數和方式
        if args.rotate_step:
            num_steps = 360 // args.rotate_step
            iterations = range(num_steps)
            print(f"將為 {filename} 產生 {num_steps} 個固定角度擴增版本...")
        else:
            iterations = range(args.num_augmentations)

        for i in iterations:
            try:
                # --- 進行擴增 ---
                if args.rotate_step:
                    angle = i * args.rotate_step
                    # 建立一個包含固定旋轉和隨機擴增的流程
                    # 這裡我們將固定旋轉與 `transform` 中的其他擴增結合
                    rotation_transform = A.Compose([
                        A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)] + transform.transforms,
                        bbox_params=transform.processors['bboxes'].params
                    )
                    augmented = rotation_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    new_filename_base = f"{os.path.splitext(filename)[0]}_rot{angle}"
                else:
                    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    new_filename_base = f"{os.path.splitext(filename)[0]}_aug_{i}"

                aug_image = augmented['image']
                aug_h, aug_w, _ = aug_image.shape
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                if not aug_bboxes: # 如果擴增後所有物件都超出邊界，則跳過
                    continue

                # 產生新的檔名
                new_image_filename = new_filename_base + '.jpg'
                new_image_path = os.path.join(args.output_images, new_image_filename)
                new_label_path = os.path.join(args.output_labels, new_filename_base + ann_ext)

                # 儲存擴增後的影像與標籤
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(new_image_path, aug_image_bgr)
                
                save_kwargs = {
                    'filename': new_image_filename,
                    'width': aug_w,
                    'height': aug_h,
                    'depth': 3
                }
                save_func(new_label_path, aug_bboxes, aug_labels, **save_kwargs)

            except Exception as e:
                print(f"錯誤：在擴增檔案 {filename} 時發生問題: {e}")

    print("影像擴增完成！")
    print(f"擴增後的影像儲存於: {args.output_images}")
    print(f"擴增後的標籤儲存於: {args.output_labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 資料集影像擴增 CLI 工具")
    parser.add_argument("--images_dir", type=str, required=True, help="原始影像資料夾的路徑")
    parser.add_argument("--labels_dir", type=str, required=True, help="原始標籤資料夾的路徑")
    parser.add_argument("--output_images", type=str, required=True, help="擴增後影像的儲存路徑")
    parser.add_argument("--output_labels", type=str, required=True, help="擴增後標籤的儲存路徑")
    parser.add_argument("--num_augmentations", type=int, default=5, help="每張原始影像要產生的擴增版本數量")
    parser.add_argument("--rotate_limit", type=int, default=None, help="(模式1) 指定隨機旋轉的最大角度 (例如: 40)。")
    parser.add_argument("--rotate_step", type=int, default=None, help="(模式2) 指定固定旋轉的角度步長 (例如: 45)。若指定此項，會忽略 --num_augmentations。")
    
    args = parser.parse_args()
    main(args)

# --images_dir D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\images --labels_dir D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\labels --output_images D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\images --output_labels D:\Dre\PDE_yolo_infra\train_dataset\hph_packing_merged_aug\labels