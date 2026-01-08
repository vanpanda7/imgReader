# JM 漫画阅读器

基于 FastAPI 的漫画阅读器，支持 WebP 格式图片浏览。

## 项目结构

```
JM/
├── api.py              # FastAPI 后端 API
├── config.py           # 配置文件
├── src/
│   └── jmcomic/        # JMComic 核心模块（本地模块）
│       ├── __init__.py
│       ├── api.py
│       ├── jm_client_impl.py
│       ├── jm_downloader.py
│       └── ...         # 其他 jmcomic 模块文件
├── static/
│   ├── viewer.html     # 前端阅读器页面
│   └── index.html      # 首页
├── option.yml          # JMComic 选项配置
├── requirements.txt    # Python 依赖
├── start_api.sh        # 启动脚本
└── README.md          # 本文件
```

## 重要说明

### jmcomic 模块

`jmcomic` 是项目的**本地模块**，位于 `src/jmcomic/` 目录下，不是通过 pip 安装的第三方包。

- `api.py` 会自动将 `src` 目录添加到 Python 路径，以便导入 `jmcomic` 模块
- 确保 `src/jmcomic/` 目录存在且完整

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**重要**：`jmcomic` 模块依赖 `commonX` 包（在 requirements.txt 中已包含），确保已正确安装。

### 2. 配置

编辑 `config.py` 文件，配置：
- Redis 连接信息
- NAS 路径配置
- 服务器端口等

### 3. 启动服务

```bash
./start_api.sh
```

或者直接运行：

```bash
python api.py
```

服务将在 `http://localhost:8000` 启动。

## 功能特性

- 📖 支持 WebP 格式图片浏览
- 🖼️ 自动生成缩略图（支持 WebP/JPEG）
- 📁 多目录支持（JM、漫画等）
- 🔄 Redis 缓存支持
- 📱 移动端适配
- ⚡ 高性能图片加载

## 配置说明

主要配置在 `config.py` 中：

- `REDIS_CONFIG`: Redis 缓存配置
- `NAS_CONFIG`: NAS 存储配置
- `SERVER_CONFIG`: 服务器配置
- `API_CONFIG`: API 相关配置（缓存、图片质量等）

## API 端点

- `GET /` - 健康检查
- `GET /viewer.html` - 阅读器页面
- `GET /api/files` - 获取文件列表
- `GET /api/image/{path}` - 获取图片
- `GET /api/thumbnail/{path}` - 获取缩略图
- `GET /api/comic-dirs` - 获取可用目录列表

## 注意事项

- 确保 NAS 路径已正确挂载
- Redis 为可选，未安装将使用内存缓存
- 图片质量配置在 `API_CONFIG["image"]["image_quality"]` 中

