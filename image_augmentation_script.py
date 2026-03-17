#!/usr/bin/env python3
"""
影像擴增腳本 (Image Augmentation Script)
功能：對影像進行多種擴增處理，並同步更新 PASCAL VOC 格式的邊界框座標

作者：AI Assistant
版本：1.0
日期：2025年7月
"""

import os
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import albumentations as A
from albumentations.core.composition import Compose


class PascalVOCProcessor:
    """處理 PASCAL VOC 格式 XML 檔案的類別"""
    
    def __init__(self):
        pass
    
    def parse_xml(self, xml_path: str) -> Dict[str, Any]:
        """
        解析 PASCAL VOC XML 檔案
        
        Args:
            xml_path (str): XML 檔案路徑
            
        Returns:
            Dict: 包含影像資訊和物件標註的字典
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 提取影像基本資訊
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = size.find('depth').text
        
        # 提取所有物件標註
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            # 提取邊界框座標
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'depth': depth,
            'objects': objects
        }
    
    def create_xml(self, annotation_data: Dict[str, Any], output_path: str) -> None:
        """
        建立新的 PASCAL VOC XML 檔案
        
        Args:
            annotation_data (Dict): 標註資料
            output_path (str): 輸出檔案路徑
        """
        # 建立根元素
        annotation = ET.Element('annotation')
        
        # 添加基本資訊
        folder = ET.SubElement(annotation, 'folder')
        folder.text = 'augmented'
        
        filename = ET.SubElement(annotation, 'filename')
        filename.text = annotation_data['filename']
        
        # 添加影像尺寸資訊
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(annotation_data['width'])
        height = ET.SubElement(size, 'height')
        height.text = str(annotation_data['height'])
        depth = ET.SubElement(size, 'depth')
        depth.text = str(annotation_data['depth'])
        
        # 添加物件標註
        for obj in annotation_data['objects']:
            object_elem = ET.SubElement(annotation, 'object')
            
            name = ET.SubElement(object_elem, 'name')
            name.text = obj['name']
            
            pose = ET.SubElement(object_elem, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(object_elem, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(object_elem, 'difficult')
            difficult.text = '0'
            
            # 添加邊界框
            bndbox = ET.SubElement(object_elem, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(obj['bbox'][0]))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(obj['bbox'][1]))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(obj['bbox'][2]))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(obj['bbox'][3]))
        
        # 寫入檔案
        tree = ET.ElementTree(annotation)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)


class ImageAugmentator:
    """影像擴增處理類別"""
    
    def __init__(self, augmentation_config: Dict[str, Any] = None):
        """
        初始化擴增器
        
        Args:
            augmentation_config (Dict): 擴增參數設定
        """
        self.config = augmentation_config or {}
        self.voc_processor = PascalVOCProcessor()
        
        # 設定預設的擴增管道
        self.augmentation_pipelines = self._create_augmentation_pipelines()
    
    def _create_augmentation_pipelines(self) -> Dict[str, Compose]:
        """
        建立不同的擴增管道
        
        Returns:
            Dict: 包含不同擴增方法的字典
        """
        pipelines = {}
        
        # 水平翻轉
        pipelines['horizontal_flip'] = A.Compose([
            A.HorizontalFlip(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        # 亮度對比度調整
        pipelines['brightness_contrast'] = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        # 色調飽和度調整
        pipelines['hue_saturation'] = A.Compose([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        # 混合擴增（同時應用多種技術）
        pipelines['mixed'] = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        return pipelines
    
    def augment_image_with_annotations(self, image_path: str, xml_path: str, 
                                     output_dir: str, augmentation_types: List[str]) -> None:
        """
        對單一影像及其標註進行擴增
        
        Args:
            image_path (str): 影像檔案路徑
            xml_path (str): XML 標註檔案路徑
            output_dir (str): 輸出資料夾路徑
            augmentation_types (List[str]): 要執行的擴增類型列表
        """
        # 讀取影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"錯誤：無法讀取影像 {image_path}")
            return
        
        # 解析 XML 標註
        try:
            annotation_data = self.voc_processor.parse_xml(xml_path)
        except Exception as e:
            print(f"錯誤：無法解析 XML 檔案 {xml_path}，錯誤訊息：{e}")
            return
        
        # 準備邊界框資料 (albumentations 格式)
        bboxes = []
        class_labels = []
        for obj in annotation_data['objects']:
            bboxes.append(obj['bbox'])
            class_labels.append(obj['name'])
        
        # 取得檔案名稱（不含副檔名）
        base_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        
        # 對每種指定的擴增方法進行處理
        for aug_type in augmentation_types:
            if aug_type not in self.augmentation_pipelines:
                print(f"警告：不支援的擴增類型 '{aug_type}'，跳過")
                continue
            
            try:
                # 執行擴增
                augmented = self.augmentation_pipelines[aug_type](
                    image=image, 
                    bboxes=bboxes, 
                    class_labels=class_labels
                )
                
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_labels = augmented['class_labels']
                
                # 如果邊界框為空（可能被裁切掉了），跳過這次擴增
                if len(augmented_bboxes) == 0:
                    print(f"警告：擴增後物件全部被裁切，跳過 {base_name}_{aug_type}")
                    continue
                
                # 更新標註資料
                updated_annotation = annotation_data.copy()
                updated_annotation['filename'] = f"{base_name}_{aug_type}{image_ext}"
                updated_annotation['height'], updated_annotation['width'] = augmented_image.shape[:2]
                
                # 更新物件標註
                updated_objects = []
                for bbox, label in zip(augmented_bboxes, augmented_labels):
                    updated_objects.append({
                        'name': label,
                        'bbox': bbox
                    })
                updated_annotation['objects'] = updated_objects
                
                # 儲存擴增後的影像
                output_image_path = os.path.join(output_dir, f"{base_name}_{aug_type}{image_ext}")
                cv2.imwrite(output_image_path, augmented_image)
                
                # 儲存更新後的 XML
                output_xml_path = os.path.join(output_dir, f"{base_name}_{aug_type}.xml")
                self.voc_processor.create_xml(updated_annotation, output_xml_path)
                
                print(f"✓ 完成擴增：{base_name}_{aug_type}")
                
            except Exception as e:
                print(f"錯誤：處理 {base_name} 的 {aug_type} 擴增時發生錯誤：{e}")
                continue
    
    def batch_augment(self, input_dir: str, output_dir: str, 
                     augmentation_types: List[str]) -> None:
        """
        批次處理資料夾中的所有影像和標註
        
        Args:
            input_dir (str): 輸入資料夾路徑
            output_dir (str): 輸出資料夾路徑
            augmentation_types (List[str]): 要執行的擴增類型列表
        """
        # 建立輸出資料夾
        os.makedirs(output_dir, exist_ok=True)
        
        # 取得所有影像檔案
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        print(f"找到 {len(image_files)} 個影像檔案")
        
        processed_count = 0
        
        for image_file in image_files:
            # 尋找對應的 XML 檔案
            xml_file = image_file.with_suffix('.xml')
            
            if not xml_file.exists():
                print(f"警告：找不到對應的 XML 檔案 {xml_file}，跳過 {image_file.name}")
                continue
            
            print(f"處理中：{image_file.name}")
            
            # 進行擴增
            self.augment_image_with_annotations(
                str(image_file), 
                str(xml_file), 
                output_dir, 
                augmentation_types
            )
            
            processed_count += 1
        
        print(f"\n處理完成！總共處理了 {processed_count} 對影像/XML 檔案")
        print(f"擴增後的檔案已儲存至：{output_dir}")


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description="影像擴增腳本 - 同步處理影像和 PASCAL VOC 標註",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            可用的擴增類型：
            horizontal_flip    - 水平翻轉
            brightness_contrast - 亮度對比度調整
            hue_saturation    - 色調飽和度調整
            mixed             - 混合擴增（同時應用多種技術）

            使用範例：
            python augment_images.py -i ./dataset -o ./augmented_dataset -a horizontal_flip rotation
            python augment_images.py -i ./data -o ./output -a mixed
        """
    )
    
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                       help='輸入資料夾路徑（包含影像和 XML 檔案）')
    
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='輸出資料夾路徑')
    
    parser.add_argument('-a', '--augmentations', nargs='+', 
                       choices=['horizontal_flip', 'brightness_contrast', 'hue_saturation', 'mixed'],
                       default=['horizontal_flip'],
                       help='要執行的擴增類型（可指定多個）')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子（確保結果可重現）')
    
    args = parser.parse_args()
    
    # 設定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(args.input_dir):
        print(f"錯誤：輸入資料夾 '{args.input_dir}' 不存在")
        return
    
    print("=" * 60)
    print("影像擴增腳本")
    print("=" * 60)
    print(f"輸入資料夾：{args.input_dir}")
    print(f"輸出資料夾：{args.output_dir}")
    print(f"擴增類型：{', '.join(args.augmentations)}")
    print(f"隨機種子：{args.seed}")
    print("=" * 60)
    
    # 建立擴增器並執行批次處理
    augmentator = ImageAugmentator()
    augmentator.batch_augment(args.input_dir, args.output_dir, args.augmentations)
    
    print("=" * 60)
    print("處理完成！")


if __name__ == "__main__":
    main()
