import os
import torch
from torchvision import transforms
from PIL import Image
import json
import pandas as pd
import time
from models.resnet_transfer import ResNet50Transfer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----- 配置 -----
MODEL_PATH = "best_model_val_6.pth"
CLASS_JSON_PATH = "class_to_idx.json"
TEST_JSON_PATH = "/root/autodl-fs/Animals/data/split_jsons/test_20.json"
ROOT_DIR = "/root/autodl-fs/Animals"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 图像预处理 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----- 加载类别映射 -----
with open(CLASS_JSON_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = list(class_to_idx.keys())

# ----- 加载模型 -----
num_classes = len(class_to_idx)
model = ResNet50Transfer(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- 单张图片预测 -----
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    return pred_idx, confidence

# ----- 主程序 -----
if __name__ == "__main__":
    with open(TEST_JSON_PATH, "r") as f:
        test_data = json.load(f)

    if not test_data["images"]:
        print("测试集为空")
        exit(0)

    correct = 0
    total = 0
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = f"test_predictions_realtime_{timestamp}.csv"

    print(f"开始处理 {len(test_data['images'])} 张测试图片...\n")

    for item in test_data["images"]:
        file_name = item["file_name"]
        image_path = os.path.join(ROOT_DIR, file_name)

        try:
            # 提取真实标签
            class_dir = file_name.split("/")[1]
            true_class_name = class_dir.split("_", 1)[1]
            true_label_idx = class_to_idx.get(true_class_name)

            if true_label_idx is None:
                raise ValueError(f"未知类别：{true_class_name}")

            pred_idx, conf = predict_image(image_path)
            pred_class = idx_to_class[pred_idx]
            is_correct = pred_idx == true_label_idx

            total += 1
            correct += int(is_correct)

            all_true_labels.append(true_class_name)
            all_pred_labels.append(pred_class)
            all_confidences.append(conf)

            print(f"[{file_name}] => 预测: {pred_class}（{conf:.2%}） | 正确: {'correct' if is_correct else 'wrong'}")

            # 实时保存到 CSV
            result_dict = {
                "file_name": file_name,
                "true_label": true_class_name,
                "predicted_label": pred_class,
                "confidence": round(conf, 4),
                "correct": is_correct
            }

            df = pd.DataFrame([result_dict])
            df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))

        except Exception as e:
            print(f"处理失败: {file_name}，错误：{e}")

    # ----- 评估指标 -----
    accuracy = 100.0 * correct / total
    avg_conf = sum(all_confidences) / len(all_confidences)

    print(f"\nAverage Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Average Confidence: {avg_conf:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = f"confusion_matrix_{timestamp}.csv"
    cm_df.to_csv(cm_csv)
    print(f"混淆矩阵已保存至：{cm_csv}")

    print(f"\n所有预测结果已实时保存至：{output_csv}")