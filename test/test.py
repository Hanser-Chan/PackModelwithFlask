import requests

url = "http://127.0.0.1:5000/api/predict"
file_path = "Z:\\memotion_dataset_7k\\images\\image_1.jpg"  # 替换为你的图片路径

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())