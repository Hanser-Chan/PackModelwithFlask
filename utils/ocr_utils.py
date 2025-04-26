import numpy as np
from paddleocr import PaddleOCR

class OcrProcessor:

    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super(OcrProcessor, cls).__new__(cls)
            print("初始化 PaddleOCR...")
            cls._instance.ocr = PaddleOCR(
                use_angle_cls=config.OCR_USE_ANGLE_CLS,
                lang=config.OCR_LANG,
                use_gpu=config.USE_GPU
            )
        return cls._instance

    def extract_text(self, image):
        """
        从图像中提取文本
        Returns:
            提取的文本字符串，提取失败则返回空字符串
        """
        try:
            # 使用PaddleOCR进行识别
            result = self.ocr.ocr(image, cls=True)

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
            return full_text
        except Exception as e:
            print(f"OCR处理出错: {str(e)}")
            return ""