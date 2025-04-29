import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 定义图像预处理转换
def get_transforms(img_size):
    """
    获取图像预处理转换

    Args:
        img_size: 图像目标尺寸 [height, width]

    Returns:
        Albumentations转换流水线
    """
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

# 定义多模态多标签模型
class MultiModalMultiLabelModel(torch.nn.Module):
    """
    多模态多标签情感分析模型
    整合了文本和图像信息，使用交叉注意力机制融合
    """
    def __init__(self, config):
        super(MultiModalMultiLabelModel, self).__init__()

        # 加载文本编码器
        self.text_encoder = AutoModel.from_pretrained(config.TOKENIZER)

        # 加载图像编码器
        self.image_encoder = AutoModel.from_pretrained(config.IMAGE_ENCODER)

        # 获取编码器输出维度
        self.text_dim = self.text_encoder.config.hidden_size
        self.image_dim = self.image_encoder.config.hidden_size

        # 全连接层
        self.text_fc = torch.nn.Linear(self.text_dim, config.CA_HIDDEN_SIZE)
        self.image_fc = torch.nn.Linear(self.image_dim, config.CA_HIDDEN_SIZE)

        # 交叉注意力模块
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=config.CA_HIDDEN_SIZE,
            num_heads=config.CA_NUM_HEADS,
            dropout=config.CA_DROPOUT,
            batch_first=True
        )

        # 分类头
        self.humour_head = torch.nn.Linear(config.CA_HIDDEN_SIZE, config.HUMOUR_CLASSES)
        self.sarcasm_head = torch.nn.Linear(config.CA_HIDDEN_SIZE, config.SARCASM_CLASSES)
        self.offensive_head = torch.nn.Linear(config.CA_HIDDEN_SIZE, config.OFFENSIVE_CLASSES)
        self.motivational_head = torch.nn.Linear(config.CA_HIDDEN_SIZE, config.MOTIVATIONAL_CLASSES)
        self.sentiment_head = torch.nn.Linear(config.CA_HIDDEN_SIZE, config.SENTIMENT_CLASSES)

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

class ModelPredictor:
    """
    模型预测器类
    单例模式，确保模型只加载一次
    """
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super(ModelPredictor, cls).__new__(cls)
            cls._instance.config = config
            cls._instance.device = config.DEVICE

            # 加载分词器
            print("加载分词器...")
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

            # 初始化并加载模型
            print("初始化并加载模型...")
            cls._instance.model = MultiModalMultiLabelModel(config)
            cls._instance.model.load_state_dict(
                torch.load(config.MODEL_PATH, map_location=config.DEVICE)
            )
            cls._instance.model.to(config.DEVICE)
            cls._instance.model.eval()

            # 准备转换器
            cls._instance.transforms = get_transforms(config.IMG_SIZE)

        return cls._instance

    def predict(self, image_array, text):
        """
        使用模型预测图像的情感标签
        Returns:
            预测结果字典
        """
        config = self.config
        device = self.device

        # 图像预处理
        image = Image.fromarray(image_array)
        pixel_values = self.transforms(image=np.array(image))['image'].unsqueeze(0).to(device)

        # 文本预处理
        encoded_text = self.tokenizer(
            text if text else "",  # 如果text为None或空，就使用空字符串
            padding='max_length',
            truncation=True,
            max_length=config.MAX_LEN,
            return_tensors='pt'
        )

        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, pixel_values)

        # 处理预测结果
        predictions = {}

        for label_name, logits in outputs.items():
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)

            # 根据标签获取类别名称和概率
            if label_name == 'humour':
                class_map = config.HUMOUR_MAP_REVERSE
            elif label_name == 'sarcasm':
                class_map = config.SARCASM_MAP_REVERSE
            elif label_name == 'offensive':
                class_map = config.OFFENSIVE_MAP_REVERSE
            elif label_name == 'motivational':
                class_map = config.MOTIVATIONAL_MAP_REVERSE
            elif label_name == 'overall_sentiment':
                class_map = config.SENTIMENT_MAP_REVERSE

            class_name = class_map[pred_class]
            class_probs = {class_map[i]: float(probs[i]) for i in range(len(probs))}

            predictions[label_name] = {
                # 原来这里是class，但使用Springboot不能很好获取(关键字)，这里改成clazz
                'clazz': class_name,
                'confidence': float(probs[pred_class]),
                'probabilities': class_probs
            }

        return predictions