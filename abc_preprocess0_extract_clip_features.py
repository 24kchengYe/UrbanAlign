"""
CLIP特征提取脚本 (可选)
如果你没有预先准备的CLIP特征,可以使用此脚本提取
"""
import torch
import torch.utils._pytree as _pytree
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("--- 环境诊断 ---")
print(f"Python 路径: {sys.executable}")
print(f"Torch 版本: {torch.__version__}")
print(f"Torch 文件位置: {torch.__file__}")
print(f"是否存在 register_pytree_node: {hasattr(_pytree, 'register_pytree_node')}")
print("----------------")

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARN] PyTorch或transformers未安装,无法提取CLIP特征")
    print("安装方法: pip install torch transformers")

# ==============================================================================
# 配置
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv(
    "PLACE_PULSE_DIR",
    r"H:\RawData13-全球街景\mit place pulse\01 Place Pluse2.0数据集\01 Place Pulse 2.0论文数据集"
)
IMAGE_DIR = os.path.join(DATA_DIR, "final_photo_dataset")

OUTPUT_DIR = os.path.join(CURRENT_DIR, "urbanalign_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clip_embeddings.npz")

# CLIP模型配置
CLIP_MODEL = "openai/clip-vit-large-patch14"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 特征提取
# ==============================================================================
def extract_clip_features():
    """提取所有图片的CLIP特征"""

    if not CLIP_AVAILABLE:
        print("[ERROR] 无法提取CLIP特征,缺少依赖库")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"[ERROR] 图片目录不存在: {IMAGE_DIR}")
        return

    print("\n" + "="*80)
    print("CLIP Feature Extraction")
    print("="*80 + "\n")

    # 1. 加载模型
    print(f"[STEP 1] 加载CLIP模型: {CLIP_MODEL}")
    print(f"  设备: {DEVICE}")

    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    print("  模型加载完成!")

    # 2. 获取所有图片
    print("\n[STEP 2] 扫描图片...")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"  发现 {len(image_files)} 张图片")

    if len(image_files) == 0:
        print("[ERROR] 未找到图片文件!")
        return

    # 3. 批量提取特征
    print("\n[STEP 3] 提取CLIP特征...")

    all_paths = []
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing"):
            batch_files = image_files[i:i+BATCH_SIZE]
            batch_images = []
            batch_paths = []

            for img_file in batch_files:
                img_path = os.path.join(IMAGE_DIR, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    batch_paths.append(img_path)
                except:
                    continue

            if len(batch_images) == 0:
                continue

            # 处理图片
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # 提取特征
            outputs = model.get_image_features(**inputs)
            embeddings = outputs.cpu().numpy()

            all_paths.extend(batch_paths)
            all_embeddings.append(embeddings)

    # 4. 合并并保存
    print("\n[STEP 4] 保存特征...")
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    np.savez(
        OUTPUT_FILE,
        paths=np.array(all_paths),
        embeddings=all_embeddings
    )

    print(f"  保存完成: {OUTPUT_FILE}")
    print(f"  特征形状: {all_embeddings.shape}")
    print(f"  维度: {all_embeddings.shape[1]}")

    # 5. 验证
    print("\n[STEP 5] 验证...")
    data = np.load(OUTPUT_FILE)
    print(f"  加载路径数: {len(data['paths'])}")
    print(f"  嵌入形状: {data['embeddings'].shape}")

    print("\n[完成] CLIP特征提取成功!")
    print(f"  输出文件: {OUTPUT_FILE}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    """主函数"""

    if not CLIP_AVAILABLE:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     CLIP特征提取工具                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ 错误: 缺少必要的依赖库                                               ║
║                                                                      ║
║ 请安装以下库:                                                        ║
║   pip install torch transformers pillow                             ║
║                                                                      ║
║ 或者跳过此步骤:                                                      ║
║   UrbanAlign可在没有CLIP特征的情况下运行(性能可能下降)               ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        return

    if os.path.exists(OUTPUT_FILE):
        print(f"\n[INFO] CLIP特征文件已存在: {OUTPUT_FILE}")
        response = input("是否重新提取? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消。")
            return

    extract_clip_features()

if __name__ == "__main__":
    main()
