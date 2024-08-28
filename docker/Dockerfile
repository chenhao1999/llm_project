# 使用官方的 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将当前目录下的所有文件复制到容器的工作目录
COPY . /app

# 使用特定版本的 scikit-learn 安装所需的 Python 库
RUN pip install --no-cache-dir pandas joblib openpyxl scikit-learn==1.3.2

# 设置运行程序的命令
CMD ["python", "model.py"]
