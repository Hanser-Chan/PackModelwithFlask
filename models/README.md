# 模型保存路径

将模型放在这个位置

需要修改`config.py`中的`MODEL_PATH`变量

```python

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "your_model_name")

```