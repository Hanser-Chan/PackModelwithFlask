import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel, AutoTokenizer
import cv2

# 导入PaddleOCR
from paddleocr import PaddleOCR

# 配置类，定义模型参数和标签映射
class Config:
    seed = 101
    model_name = "vit-sbert-multimodal-multilabel"
    backbone = "google/vit-base-patch16-224+sentence-transformers/all-mpnet-base-v2-ep10"
    tokenizer = "sentence-transformers/all-mpnet-base-v2"
    image_encoder = "google/vit-base-patch16-224"
    img_size = [224, 224]
    max_len = 128
    
    # 多标签配置
    label_names = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
    humour_classes = 4       # not_funny, funny, very_funny, hilarious
    sarcasm_classes = 4      # not_sarcastic, general, twisted_meaning, very_twisted
    offensive_classes = 3    # not_offensive, slight, very_offensive
    motivational_classes = 2 # not_motivational, motivational
    sentiment_classes = 5    # very_negative, negative, neutral, positive, very_positive
    
    # 标签映射（反向映射，用于输出结果）
    humour_map_reverse = {0: 'not_funny', 1: 'funny', 2: 'very_funny', 3: 'hilarious'}
    sarcasm_map_reverse = {0: 'not_sarcastic', 1: 'general', 2: 'twisted_meaning', 3: 'very_twisted'}
    offensive_map_reverse = {0: 'not_offensive', 1: 'slight', 2: 'very_offensive'}
    motivational_map_reverse = {0: 'not_motivational', 1: 'motivational'}
    sentiment_map_reverse = {0: 'very_negative', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'very_positive'}
    
    # 交叉注意力参数
    ca_hidden_size = 256
    ca_num_heads = 8
    ca_dropout = 0.1
    
    # PaddleOCR配置
    use_gpu = torch.cuda.is_available()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像预处理转换
def get_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

# 定义多模态多标签模型 - 修改为与训练时相同的结构
class MultiModalMultiLabelModel(torch.nn.Module):
    def __init__(self, config):
        super(MultiModalMultiLabelModel, self).__init__()
        
        # 加载文本编码器
        self.text_encoder = AutoModel.from_pretrained(config.tokenizer)
        
        # 加载图像编码器
        self.image_encoder = AutoModel.from_pretrained(config.image_encoder)
        
        # 获取编码器输出维度
        self.text_dim = self.text_encoder.config.hidden_size
        self.image_dim = self.image_encoder.config.hidden_size
        
        # 全连接层 - 与训练时的模型结构匹配
        self.text_fc = torch.nn.Linear(self.text_dim, config.ca_hidden_size)
        self.image_fc = torch.nn.Linear(self.image_dim, config.ca_hidden_size)
        
        # 单个交叉注意力模块 - 与训练时的模型结构匹配
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=config.ca_hidden_size,
            num_heads=config.ca_num_heads,
            dropout=config.ca_dropout,
            batch_first=True
        )
        
        # 分类头
        self.humour_head = torch.nn.Linear(config.ca_hidden_size, config.humour_classes)
        self.sarcasm_head = torch.nn.Linear(config.ca_hidden_size, config.sarcasm_classes)
        self.offensive_head = torch.nn.Linear(config.ca_hidden_size, config.offensive_classes)
        self.motivational_head = torch.nn.Linear(config.ca_hidden_size, config.motivational_classes)
        self.sentiment_head = torch.nn.Linear(config.ca_hidden_size, config.sentiment_classes)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        # 文本编码
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # 使用[CLS]令牌
        
        # 图像编码
        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True
        )
        image_embeddings = image_outputs.last_hidden_state[:, 0, :]  # 使用[CLS]令牌
        
        # 投影到相同的维度空间
        text_features = self.text_fc(text_embeddings).unsqueeze(1)  # [batch_size, 1, hidden_size]
        image_features = self.image_fc(image_embeddings).unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 合并特征
        combined_features = torch.cat([text_features, image_features], dim=1)  # [batch_size, 2, hidden_size]
        
        # 自注意力
        attended_features, _ = self.cross_attention(
            query=combined_features,
            key=combined_features,
            value=combined_features
        )
        
        # 池化 - 使用第一个位置的特征（对应于文本特征）
        fused_embedding = attended_features[:, 0, :]
        
        # 多标签分类
        humour_logits = self.humour_head(fused_embedding)
        sarcasm_logits = self.sarcasm_head(fused_embedding)
        offensive_logits = self.offensive_head(fused_embedding)
        motivational_logits = self.motivational_head(fused_embedding)
        sentiment_logits = self.sentiment_head(fused_embedding)
        
        return {
            'humour': humour_logits,
            'sarcasm': sarcasm_logits,
            'offensive': offensive_logits,
            'motivational': motivational_logits,
            'overall_sentiment': sentiment_logits
        }

# OCR函数 - 使用PaddleOCR提取图像中的文本
def extract_text_from_image(image_path, config):
    """
    使用PaddleOCR从图像中提取文本
    
    参数:
        image_path: 图像文件路径
        config: 配置对象
    
    返回:
        提取的文本字符串
    """
    try:
        # 初始化PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=config.use_gpu)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return ""
        
        # 使用PaddleOCR进行识别
        result = ocr.ocr(image_path, cls=True)
        
        # 提取文本
        texts = []
        if result:
            for line in result:
                for item in line:
                    if len(item) >= 2 and isinstance(item[1], tuple) and len(item[1]) >= 1:
                        text = item[1][0]
                        texts.append(text)
        
        # 合并文本
        full_text = " ".join(texts)
        
        print(f"PaddleOCR提取的文本: {full_text}")
        return full_text
    except Exception as e:
        print(f"OCR处理时出错: {str(e)}")
        return ""

# 模型预测函数
def predict_meme(model, tokenizer, image_path, text=None, config=None):
    """
    使用训练好的模型预测表情包的多标签情感
    
    参数:
        model: 加载的模型
        tokenizer: 文本分词器
        image_path: 表情包图像路径
        text: 表情包中的文本 (如果为None，将使用OCR提取)
        config: 配置对象
    
    返回:
        预测结果字典
    """
    if config is None:
        config = Config()
    
    model.eval()
    device = config.device
    
    # 如果没有提供文本，使用PaddleOCR提取
    if text is None or text == "":
        text = extract_text_from_image(image_path, config)
        print(f"使用PaddleOCR提取的文本: {text}")
    
    # 图像预处理
    image = Image.open(image_path).convert('RGB')
    transforms = get_transforms(config.img_size)
    pixel_values = transforms(image=np.array(image))['image'].unsqueeze(0).to(device)
    
    # 文本预处理
    encoded_text = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=config.max_len,
        return_tensors='pt'
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, pixel_values)
    
    # 处理预测结果
    predictions = {}
    
    for label_name, logits in outputs.items():
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        
        # 获取类别名称
        if label_name == 'humour':
            class_name = config.humour_map_reverse[pred_class]
            class_probs = {config.humour_map_reverse[i]: float(probs[i]) for i in range(len(probs))}
        elif label_name == 'sarcasm':
            class_name = config.sarcasm_map_reverse[pred_class]
            class_probs = {config.sarcasm_map_reverse[i]: float(probs[i]) for i in range(len(probs))}
        elif label_name == 'offensive':
            class_name = config.offensive_map_reverse[pred_class]
            class_probs = {config.offensive_map_reverse[i]: float(probs[i]) for i in range(len(probs))}
        elif label_name == 'motivational':
            class_name = config.motivational_map_reverse[pred_class]
            class_probs = {config.motivational_map_reverse[i]: float(probs[i]) for i in range(len(probs))}
        elif label_name == 'overall_sentiment':
            class_name = config.sentiment_map_reverse[pred_class]
            class_probs = {config.sentiment_map_reverse[i]: float(probs[i]) for i in range(len(probs))}
        
        predictions[label_name] = {
            'class': class_name,
            'confidence': float(probs[pred_class]),
            'probabilities': class_probs
        }
    
    return predictions

# 主函数
def main():
    # 配置
    config = Config()
    
    # 检查PaddleOCR是否已安装
    try:
        import paddleocr
        print("PaddleOCR已安装")
    except ImportError:
        print("警告: 未安装PaddleOCR")
        print("请使用以下命令安装PaddleOCR:")
        print("pip install paddlepaddle")
        print("pip install paddleocr")
        
        # 询问用户是否继续
        response = input("是否继续而不使用OCR功能? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 模型路径
    model_path = "E:\\DL\\NLPtest\\example\\models\\best\\best_epoch-fold00.bin"  # 修改为您的模型文件路径
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在")
        return
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    
    # 初始化模型
    print("初始化模型...")
    model = MultiModalMultiLabelModel(config)
    
    # 加载模型权重
    print(f"从 {model_path} 加载模型权重...")
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    
    # 测试图像路径
    test_image_path = input("请输入表情包图像路径: ")
    if not os.path.exists(test_image_path):
        print(f"错误: 图像文件 '{test_image_path}' 不存在")
        return
    
    # 询问是否使用OCR
    use_ocr = input("是否使用OCR自动提取图像中的文本? (y/n): ").lower() == 'y'
    
    # 可选的表情包文本
    test_text = None
    if not use_ocr:
        test_text = input("请输入表情包中的文本 (如果没有，请直接按回车): ")
    
    # 进行预测
    print("正在预测...")
    predictions = predict_meme(model, tokenizer, test_image_path, test_text, config)
    
    # 打印预测结果
    print("\n预测结果:")
    print("=" * 50)
    for label, result in predictions.items():
        print(f"{label}:")
        print(f"  类别: {result['class']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print("  各类别概率:")
        for class_name, prob in result['probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
        print("-" * 50)
    
    # 返回JSON格式的结果示例
    import json
    print("\nJSON格式输出示例 (可用于Web API):")
    print(json.dumps(predictions, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()