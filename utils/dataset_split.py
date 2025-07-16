import os
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 设置随机种子保证可复现
random.seed(42)

# 原始 JSON 路径
TRAIN_JSON_PATH = "D:/OneDrive/online_research/wk10_inat2021_classification/json_file/0-99/train_mini_subset_00000_00099.json"
VAL_JSON_PATH = "D:/OneDrive/online_research/wk10_inat2021_classification/json_file/0-99/val_subset_00000_00099.json"

# 输出路径
OUTPUT_DIR = "D:/OneDrive/online_research/wk10_inat2021_classification/json_file/0-99"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 JSON 文件
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# 合并两个 JSON 文件
train_data = load_json(TRAIN_JSON_PATH)
val_data = load_json(VAL_JSON_PATH)
all_images = train_data['images'] + val_data['images']

# 提取类别和样本
class_to_images = defaultdict(list)
for item in all_images:
    file_path = item['file_name']
    if '/' in file_path:
        class_name = file_path.split('/')[1]
        class_to_images[class_name].append(item)

# 分层划分
train_list, val_list, test_list = [], [], []

for class_name, images in class_to_images.items():
    if len(images) < 3:
        # 类别样本太少，全部放入训练集
        train_list.extend(images)
        continue

    # 先划分 train(60%) 和 temp(40%)
    train_imgs, temp_imgs = train_test_split(
        images, test_size=0.4, random_state=42
    )
    # 再将 temp 划分为 val(20%) 和 test(20%)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=42
    )

    train_list.extend(train_imgs)
    val_list.extend(val_imgs)
    test_list.extend(test_imgs)

# 保存为 JSON 格式
def save_json(images, path):
    with open(path, 'w') as f:
        json.dump({'images': images}, f, indent=4)

save_json(train_list, os.path.join(OUTPUT_DIR, 'train_60.json'))
save_json(val_list, os.path.join(OUTPUT_DIR, 'val_20.json'))
save_json(test_list, os.path.join(OUTPUT_DIR, 'test_20.json'))

print(f"数据划分完成，共 {len(train_list)} 训练，{len(val_list)} 验证，{len(test_list)} 测试样本")