import json

# 编号范围
START_ID = 0
END_ID = 1000

# 路径（Windows）
ORIGINAL_JSON = r"D:\OneDrive\online_research\wk10\inat2021\train_mini.json"
NEW_JSON = r"D:\OneDrive\online_research\wk10\inat2021\train_mini_subset_00000_01000.json"

def is_in_range_from_file_name(file_name):
    try:
        folder_name = file_name.split("/")[1]
        prefix = int(folder_name.split("_")[0])
        return START_ID <= prefix <= END_ID
    except:
        return False

def extract_subset_images_only(original_json_path, output_json_path):
    with open(original_json_path, "r") as f:
        data = json.load(f)

    # 只筛选 images，其它信息保留
    selected_images = [
        img for img in data['images']
        if is_in_range_from_file_name(img['file_name'])
    ]

    print(f"保留图像数: {len(selected_images)}（原始: {len(data['images'])}）")
    print(f"annotations 和 categories 不变")

    # 构建新 JSON：只替换 images
    new_data = data.copy()
    new_data['images'] = selected_images

    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"\n新的 JSON 文件已保存到:\n{output_json_path}")

if __name__ == "__main__":
    extract_subset_images_only(ORIGINAL_JSON, NEW_JSON)