import argparse
import cv2
import albumentations as A
from pathlib import Path
 
def get_image_only_pipeline():
    """建立純影像擴增管道 (不含標註框參數)"""
    transforms = [
        A.SafeRotate(limit=45, p=0.6, border_mode=cv2.BORDER_CONSTANT),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.15), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1.0, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(0.025, 0.075),
            hole_width_range=(0.025, 0.075),
            fill="random_uniform",
            p=0.3
        ),
    ]
    # 直接回傳 transforms，不需要 bbox_params
    return A.Compose(transforms)
 
def process_image_only(img_path, output_dir, pipeline, prefix="aug_"):
    """處理單張影像的擴增"""
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    # 讀取影像並轉換色彩空間
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"[警告] 無法讀取影像: {img_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
 
    # 執行純影像擴增 (只需傳入 image)
    try:
        transformed = pipeline(image=image_rgb)
        aug_image_rgb = transformed['image']
    except Exception as e:
        print(f"[錯誤] 處理 {img_path.name} 時發生例外狀況: {e}")
        return
 
    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # 儲存擴增後的影像
    aug_img_name = f"{prefix}{img_path.name}"
    aug_img_path = output_dir / aug_img_name
    aug_image_bgr = cv2.cvtColor(aug_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(aug_img_path), aug_image_bgr)
    print(f"[成功] 已產生: {aug_img_name}")
 
def main():
    parser = argparse.ArgumentParser(description="純影像批次擴增工具")
    parser.add_argument("-i", "--input", type=str, required=True, help="輸入的路徑 (單一圖片檔案或包含圖片的資料夾)")
    parser.add_argument("-o", "--output", type=str, default="./augmented_output", help="輸出的資料夾路徑")
    parser.add_argument("-n", "--num_aug", type=int, default=1, help="每張圖片要產生幾種擴增版本")
    args = parser.parse_args()
    input_path = Path(args.input)
    pipeline = get_image_only_pipeline()
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
 
    if input_path.is_file():
        if input_path.suffix.lower() in valid_extensions:
            for i in range(args.num_aug):
                prefix = f"aug_{i+1}_" if args.num_aug > 1 else "aug_"
                process_image_only(input_path, args.output, pipeline, prefix)
        else:
            print(f"[錯誤] {input_path} 不是支援的影像格式。")
    elif input_path.is_dir():
        img_files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions]
        if not img_files:
            print(f"[提示] 在資料夾 {input_path} 中沒有找到支援的影像檔案。")
            return
        print(f"找到 {len(img_files)} 張影像，開始處理...")
        for img_file in img_files:
            for i in range(args.num_aug):
                prefix = f"aug_{i+1}_" if args.num_aug > 1 else "aug_"
                process_image_only(img_file, args.output, pipeline, prefix)
    else:
        print("[錯誤] 輸入路徑無效，找不到檔案或資料夾。")
 
if __name__ == "__main__":
    main()