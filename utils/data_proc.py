import os
import tarfile
import json
from tqdm import tqdm

# 目录配置
DATA_DIR = "D:\OneDrive\online_research\wk10\inat2021"
TAR_PATH = os.path.join(DATA_DIR, "train_mini.tar.gz")
JSON_TAR_PATH = os.path.join(DATA_DIR, "train_mini.json.tar.gz")
EXTRACTED_JSON_PATH = os.path.join(DATA_DIR, "train_mini.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "animals_mini")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_tar_gz(tar_path, category_dirs, extract_to):
    """
    解压 tar.gz 文件中属于指定目录的文件，支持断点续解压。
    """
    category_set = set(category_dirs)

    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tqdm(tar, desc="解压 Animalia 图像（支持断点续传）"):
            path_parts = member.name.split("/")
            if len(path_parts) < 2:
                continue

            top_dir = path_parts[1]
            if top_dir in category_set:

                # 构造相对路径并排除目录项
                relative_path = "/".join(path_parts[1:])
                target_path = os.path.join(extract_to, relative_path)

                if member.isdir():
                    continue  # 跳过目录

                if os.path.exists(target_path):
                    continue  # 已存在则跳过

                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                except OSError as e:
                    print(f"创建目录失败: {target_path}，原因: {e}")
                    continue

                try:
                    source = tar.extractfile(member)
                    if source is None:
                        print(f"跳过无效文件: {member.name}")
                        continue
                    with open(target_path, "wb") as dest:
                        dest.write(source.read())
                except Exception as e:
                    print(f"解压失败: {member.name}，原因: {e}")

def get_animal_image_dirs_from_categories(json_path):
    """
    从 JSON 中找出所有 kingdom == "Animalia" 的类别，
    并返回它们的 image_dir_name。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    animal_dirs = set()

    for cat in data['categories']:
        if cat.get("kingdom", "").strip().lower() == "animalia":
            dir_name = cat.get("image_dir_name", "").strip()
            if dir_name:
                animal_dirs.add(dir_name)

    print(f"找到 {len(animal_dirs)} 个动物类目录")
    return list(animal_dirs)

def main():
    # 解压 JSON 文件（如果尚未解压）
    if not os.path.exists(EXTRACTED_JSON_PATH):
        print("正在解压 JSON 文件...")
        with tarfile.open(JSON_TAR_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
    else:
        print("JSON 文件已存在，跳过解压。")

    # 获取动物类 image_dir_name
    animal_dirs = get_animal_image_dirs_from_categories(EXTRACTED_JSON_PATH)
    print("示例目录名:", animal_dirs[:5])

    # 解压图像到目标目录
    print("开始解压动物图像...")
    extract_tar_gz(TAR_PATH, animal_dirs, OUTPUT_DIR)
    print("解压完成，文件已保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()