import torch
import os

class Config:
    # 基本配置
    SEED = 101
    MODEL_NAME = "vit-sbert-multimodal-multilabel"

    # 路径配置
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_epoch-fold00.bin")

    # 编码器配置
    BACKBONE = "google/vit-base-patch16-224+sentence-transformers/all-mpnet-base-v2-ep10"
    TOKENIZER = "sentence-transformers/all-mpnet-base-v2"
    IMAGE_ENCODER = "google/vit-base-patch16-224"

    # 图像和文本处理配置
    IMG_SIZE = [224, 224]
    MAX_LEN = 128

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 多标签配置
    LABEL_NAMES = ['humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']
    HUMOUR_CLASSES = 4       # not_funny, funny, very_funny, hilarious
    SARCASM_CLASSES = 4      # not_sarcastic, general, twisted_meaning, very_twisted
    OFFENSIVE_CLASSES = 3    # not_offensive, slight, very_offensive
    MOTIVATIONAL_CLASSES = 2 # not_motivational, motivational
    SENTIMENT_CLASSES = 5    # very_negative, negative, neutral, positive, very_positive

    # 标签映射
    HUMOUR_MAP_REVERSE = {0: 'not_funny', 1: 'funny', 2: 'very_funny', 3: 'hilarious'}
    SARCASM_MAP_REVERSE = {0: 'not_sarcastic', 1: 'general', 2: 'twisted_meaning', 3: 'very_twisted'}
    OFFENSIVE_MAP_REVERSE = {0: 'not_offensive', 1: 'slight', 2: 'very_offensive'}
    MOTIVATIONAL_MAP_REVERSE = {0: 'not_motivational', 1: 'motivational'}
    SENTIMENT_MAP_REVERSE = {0: 'very_negative', 1: 'negative', 2: 'neutral', 3: 'positive', 4: 'very_positive'}

    # 交叉注意力参数
    CA_HIDDEN_SIZE = 256
    CA_NUM_HEADS = 8
    CA_DROPOUT = 0.1

    # PaddleOCR配置
    USE_GPU = torch.cuda.is_available()
    OCR_LANG = "ch"  # 中文识别
    OCR_USE_ANGLE_CLS = True  # 使用角度分类器