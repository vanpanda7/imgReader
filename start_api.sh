#!/bin/bash
# 启动 FastAPI 服务器

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 设置 PYTHONPATH，让 Python 能找到 src/jmcomic 模块
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "启动 FastAPI 服务器..."
python api.py

