"""
统一配置文件
"""
import os
from pathlib import Path

# ==================== Redis 配置 ====================
REDIS_CONFIG = {
    "host": "192.168.30.235",
    "port": 16379,
    "password": "redis_yBZkNh",
    "db": 5,
    "decode_responses": True,
    "socket_connect_timeout": 5,
    "socket_timeout": 5,
}

# ==================== NAS 配置 ====================
NAS_CONFIG = {
    "enabled": True,
    "type": "webdav",  # smb, nfs, webdav
    "mount_point": "/mnt/nas",
    "share_path": "/home/rick/hub/漫画/JM",  # 默认路径
    # 可用的漫画目录列表
    "comic_dirs": [
        {
            "name": "JM",
            "path": "/home/rick/hub/漫画/JM",
            "display_name": "JM"
        },
        {
            "name": "漫画",
            "path": "/home/rick/hub/漫画/日漫",
            "display_name": "漫画"
        }
    ],
    
    # WebDAV 配置
    "webdav": {
        "url": "http://192.168.30.8:5005/",
        "username": "rick",
        "password": "12345678hH",
        "options": "uid=$(id -u),gid=$(id -g),file_mode=0666,dir_mode=0777",
    },
    
    # SMB/CIFS 配置
    "smb": {
        "server": "192.168.30.8",
        "username": "rick",
        "password": "12345678hH",
        "domain": "",
    },
    
    # NFS 配置
    "nfs": {
        "server": "192.168.30.8",
        "version": "4",
        "options": "rw,sync,hard,intr",
    },
}

# ==================== 服务器配置 ====================
SERVER_CONFIG = {
    "host": "::",
    "port": 8000,
}

# ==================== 应用配置 ====================
APP_CONFIG = {
    "static_dir": os.path.join(os.path.dirname(__file__), "static"),
    "option_file": os.path.join(os.path.dirname(__file__), "option.yml"),
}

# ==================== API 配置 ====================
API_CONFIG = {
    # 固定NAS路径（如果指定了固定路径，优先使用此路径）
    "fixed_nas_path": "/mnt/nas/home/hub/漫画",
    
    # 缓存配置
    "cache": {
        # 普通缓存TTL（秒），默认5分钟
        "default_ttl": 300,
        # 缩略图缓存TTL（秒），默认1小时
        "thumbnail_ttl": 3600,
        # 内存缓存最大条目数，超过后自动清理最旧的条目
        "memory_cache_max_size": 1000,
        # Redis扫描时的批量大小
        "redis_scan_count": 100,
    },
    
    # 图片配置
    "image": {
        # 图片强缓存时间（秒），默认1年
        "cache_max_age": 31536000,
        # 缩略图缓存时间（秒），默认1小时
        "thumbnail_cache_max_age": 3600,
        # 图片压缩质量（1-100），值越大质量越高但文件越大
        # 适用于JPEG和WebP格式，WebP格式也支持质量参数
        "image_quality": 90,
        # 缩略图输出格式：'auto'（自动检测源格式）、'webp'（强制WebP）、'jpeg'（强制JPEG）
        # 'auto'模式：如果源图片是WebP则输出WebP，否则输出JPEG
        "thumbnail_format": "auto",
        # 默认缩略图宽度（像素）
        "thumbnail_default_width": 200,
        # 默认缩略图高度（像素）
        "thumbnail_default_height": 200,
    },
    
    # 分页配置
    "pagination": {
        # 默认每页显示数量
        "default_page_size": 20,
    },
    
    # 线程池配置
    "thread_pool": {
        # 下载管理器线程池最大工作线程数
        "max_workers": 2,
    },
    
    # 压缩配置
    "compression": {
        # GZip中间件最小压缩大小（字节），小于此大小的响应不压缩
        "gzip_minimum_size": 1000,
    },
}
