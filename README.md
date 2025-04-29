# 使用Flask API 打包模型

## 功能

将训练好的情感分析模型制作成API，可提供表情包情感多标签分析功能，可以分析表情包的五个情感维度（幽默、讽刺、冒犯、激励和整体情感），并返回OCR识别的文本内容。

## 端点

### 情感分析接口

**URL:** `/api/predict`  
**方法:** POST  
**输入格式:** form-data  
**参数:**
- `file`: 图片文件（支持png, jpg, jpeg, gif格式）

**响应格式:** JSON

成功响应，模型推理返回示例:
```json
{
  "status": "success",
  "ocr_text": "图片中识别的文本内容",
  "predictions": {
    "humour": {
      "clazz": "funny",
      "confidence": 0.85,
      "probabilities": {
        "not_funny": 0.1,
        "funny": 0.85, 
        "very_funny": 0.03, 
        "hilarious": 0.02
      }
    },
    "sarcasm": {
      "clazz": "general",
      "confidence": 0.72,
      "probabilities": {...}  //同上显示不同值的概率分布
    },
    "offensive": {
      "clazz": "not_offensive",
      "confidence": 0.91,
      "probabilities": {...}
    },
    "motivational": {
      "clazz": "not_motivational",
      "confidence": 0.67,
      "probabilities": {...}
    },
    "overall_sentiment": {
      "clazz": "positive",
      "confidence": 0.78,
      "probabilities": {...}
    }
  }
}
```

错误响应示例:
```json
{
  "status": "error",
  "message": "错误描述"
}
```

### 健康检查接口

**URL:** `/api/health`  
**方法:** GET  

**响应格式:** JSON
```json
{
  "status": "ok",
  "message": "服务运行正常"
}
```

## 测试方法

### curl

```bash
curl -X POST -F "file=@/path/to/image.jpg" http://127.0.0.1:5000/api/predict
```

### Python脚本

可参考`test/test.py`

### Postman

1. 创建新的POST请求，URL设为`http://127.0.0.1:5000/api/predict`
2. 在Body标签页选择`form-data`
3. 添加key为`file`的字段，类型选择File，然后上传测试图片
4. 点击Send发送请求，可查看返回结果

## 错误码

- 400: 请求参数错误（文件缺失、格式不支持等）
- 500: 服务器内部错误（模型处理异常等）

