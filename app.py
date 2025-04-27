from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import time
import uuid

from config import Config
from utils.model_utils import ModelPredictor
from utils.ocr_utils import OcrProcessor

app = Flask(__name__)

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 创建临时文件目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 验证文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化模型和OCR处理器
config = Config()
model_predictor = ModelPredictor(config)  # 单例模式，模型只加载一次
ocr_processor = OcrProcessor(config)  # 单例模式，OCR引擎只初始化一次

@app.route('/api/predict', methods=['POST'])
def predict_image():
    """
    处理图片上传并返回情感分析和OCR结果
    """
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '没有找到文件'
        }), 400

    file = request.files['file']

    # 检查文件是否存在且是允许的类型
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': '没有选择文件'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': f'不支持的文件类型，请上传 {", ".join(ALLOWED_EXTENSIONS)} 格式的图片'
        }), 400

    try:
        # 读取文件
        filestr = file.read()

        # 转换为OpenCV格式
        nparr = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({
                'status': 'error',
                'message': '无法解析图像数据'
            }), 400

        # 从图像中提取文本
        extracted_text = ocr_processor.extract_text(image)

        # 使用模型进行预测
        predictions = model_predictor.predict(image, extracted_text)

        # 返回预测结果和OCR文本
        return jsonify({
            'status': 'success',
            'ocr_text': extracted_text,
            'predictions': predictions
        })

    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理图像时出错: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    API健康检查端点
    """
    return jsonify({
        'status': 'ok',
        'message': '服务运行正常'
    })

# 启动应用
if __name__ == '__main__':
    # 应用启动前确保目录存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    print(f"模型已加载，运行在 {config.DEVICE}")
    app.run(host='0.0.0.0', port=5000, debug=False)
