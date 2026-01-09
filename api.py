import os
import shutil
import base64
import hashlib
import json
import asyncio
import threading
import re
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Set, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from collections import defaultdict

# 引入 FastAPI 相关组件
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# 引入异步 Redis (替代同步 redis)
try:
    from redis import asyncio as aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    AIOREDIS_AVAILABLE = False
    print("⚠ aioredis 未安装，将使用内存缓存。请运行: pip install redis>=4.2.0")

# 引入项目依赖
# 将 src 目录添加到 Python 路径，以便导入 jmcomic 模块
# 这样即使不设置 PYTHONPATH 也能正常工作
import sys
from pathlib import Path
_src_path = Path(__file__).parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import jmcomic
import config

# ==================== 配置与初始化 ====================

# 从配置文件读取配置
NAS_CONFIG = config.NAS_CONFIG
APP_CONFIG = config.APP_CONFIG
SERVER_CONFIG = config.SERVER_CONFIG
REDIS_CONFIG = config.REDIS_CONFIG
API_CONFIG = config.API_CONFIG

# 定义独立的图片处理线程池 (建议设置为 CPU 核心数，例如 4)
# 这样可以保证看图时不影响目录加载
import os
cpu_count = os.cpu_count() or 4
image_executor = ThreadPoolExecutor(max_workers=min(cpu_count, 4), thread_name_prefix="img_worker")
print(f"✓ 图片处理线程池已创建 (工作线程数: {min(cpu_count, 4)})")

# ==================== 核心修改区域 Start ====================

# 1. 从配置文件读取固定NAS路径
FIXED_NAS_PATH = Path(API_CONFIG.get("fixed_nas_path", "/mnt/nas/home/hub/漫画"))

def get_default_image_dir() -> Path:
    """获取默认图片根目录 (修正为 /mnt/nas/home/hub/漫画)"""
    
    # 1. 优先策略：直接检查你指定的绝对路径
    if FIXED_NAS_PATH.exists():
        print(f"✓ 成功锁定 NAS 路径: {FIXED_NAS_PATH}")
        return FIXED_NAS_PATH.resolve()
    
    # 调试信息：如果上面没进去，说明这个路径在系统看来不存在
    # 这时候很有可能是挂载没成功，或者 Docker 容器没映射对
    print(f"DEBUG: 尝试访问路径 {FIXED_NAS_PATH} 失败，检查是否存在或有权限")

    # 2. 备用策略：尝试通过配置文件拼接（以此兜底）
    if NAS_CONFIG.get('enabled'):
        nas_mount = Path(NAS_CONFIG['mount_point'])
        share_path = NAS_CONFIG.get('share_path', '')
        
        # 你的情况特殊，如果是绝对路径直接用
        target = Path(share_path) if Path(share_path).is_absolute() else nas_mount / share_path.lstrip('/')
        
        # 尝试创建
        if not target.exists():
            try:
                target.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠ 创建备用路径失败: {e}")
        
        return target.resolve()

    # 3. 本地回退逻辑
    local = Path(__file__).parent / "image"
    local.mkdir(exist_ok=True)
    return local.resolve()

DEFAULT_IMAGE_DIR = get_default_image_dir()
# ==================== 核心修改区域 End ====================

# ==================== 核心服务类 ====================

class CacheService:
    """统一管理 Redis (异步) 和内存缓存"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_cache()
        return cls._instance
    
    async def _init_cache_async(self):
        """异步初始化 Redis 连接（需要在事件循环中调用）"""
        self.redis = None
        try:
            # 使用 aioredis 创建异步连接池
            if AIOREDIS_AVAILABLE and REDIS_CONFIG:
                # 构建 Redis URL
                host = REDIS_CONFIG.get('host', 'localhost')
                port = REDIS_CONFIG.get('port', 6379)
                password = REDIS_CONFIG.get('password')
                db = REDIS_CONFIG.get('db', 0)
                
                # 构建连接参数
                if password:
                    redis_url = f"redis://:{password}@{host}:{port}/{db}"
                else:
                    redis_url = f"redis://{host}:{port}/{db}"
                
                # 使用 aioredis.from_url 创建异步连接
                self.redis = await aioredis.from_url(
                    redis_url,
                    decode_responses=False  # 保持为 bytes，方便处理图片
                )
                # 测试连接
                await self.redis.ping()
                print("✓ Redis (Async) 连接成功")
        except Exception as e:
            print(f"⚠ Redis 连接失败: {e}，将使用内存缓存")
            self.redis = None

    def _init_cache(self):
        """同步初始化（仅初始化内存缓存，Redis 异步初始化在启动时完成）"""
        self.redis = None  # 将在启动时异步初始化
        self.memory_cache = {}
        self.mem_lock = threading.Lock()  # 内存缓存依然用锁
        self._redis_initialized = False  # 标记 Redis 是否已初始化
    
    async def _ensure_redis(self):
        """确保 Redis 连接已初始化（延迟初始化）"""
        if self._redis_initialized:
            return
        
        if self.redis is not None:
            self._redis_initialized = True
            return
            
        try:
            if AIOREDIS_AVAILABLE and REDIS_CONFIG:
                host = REDIS_CONFIG.get('host', 'localhost')
                port = REDIS_CONFIG.get('port', 6379)
                password = REDIS_CONFIG.get('password')
                db = REDIS_CONFIG.get('db', 0)
                
                # 构建连接参数
                if password:
                    redis_url = f"redis://:{password}@{host}:{port}/{db}"
                else:
                    redis_url = f"redis://{host}:{port}/{db}"
                
                # 使用 aioredis.from_url 创建异步连接
                self.redis = await aioredis.from_url(
                    redis_url,
                    decode_responses=False  # 保持为 bytes，方便处理图片
                )
                # 测试连接
                await self.redis.ping()
                self._redis_initialized = True
                print("✓ Redis (Async) 连接成功")
        except Exception as e:
            print(f"⚠ Redis 连接失败: {e}，将使用内存缓存")
            self.redis = None
            self._redis_initialized = True  # 标记为已尝试，避免重复尝试
    
    def get_key(self, prefix: str, *args) -> str:
        key_str = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"jmcomic:{prefix}:{key_hash}"
    
    async def get(self, key: str):
        # 确保 Redis 已初始化
        await self._ensure_redis()
        
        # 1. 尝试 Redis (异步)
        if self.redis:
            try:
                # 重点：加了 await
                cached = await self.redis.get(key)
                if cached: 
                    # 只有 JSON 字符串才 decode，否则如果是 bytes 直接处理会报错
                    # 这里假设普通 get 存的是 JSON 字符串
                    return json.loads(cached.decode('utf-8'))
            except Exception: pass
            
        # 2. 降级到内存缓存
        with self.mem_lock:
            return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = None):
        # 确保 Redis 已初始化
        await self._ensure_redis()
        
        if ttl is None:
            ttl = API_CONFIG["cache"]["default_ttl"]
            
        # 序列化
        val_str = json.dumps(value, ensure_ascii=False, default=str)
        
        # 1. 尝试 Redis (异步)
        if self.redis:
            try:
                # 重点：加了 await
                await self.redis.setex(key, ttl, val_str)
                return
            except Exception: pass
            
        # 2. 内存缓存
        with self.mem_lock:
            self.memory_cache[key] = value
            max_size = API_CONFIG["cache"]["memory_cache_max_size"]
            if len(self.memory_cache) > max_size:
                self.memory_cache.pop(next(iter(self.memory_cache)))

    async def get_thumbnail(self, key: str) -> Optional[Tuple[bytes, str]]:
        """获取二进制缓存 (异步)"""
        # 确保 Redis 已初始化
        await self._ensure_redis()
        
        if self.redis:
            try:
                # 重点：并发获取 data 和 mime，提高速度
                data, mime = await asyncio.gather(
                    self.redis.get(f"{key}:data"),
                    self.redis.get(f"{key}:mime")
                )
                if data and mime:
                    return (data, mime.decode('utf-8'))
            except Exception: pass
        return None

    async def set_thumbnail(self, key: str, data: bytes, mime_type: str, ttl: int = None):
        """设置二进制缓存 (异步)"""
        # 确保 Redis 已初始化
        await self._ensure_redis()
        
        if ttl is None:
            ttl = API_CONFIG["cache"]["thumbnail_ttl"]
        if self.redis:
            try:
                # 重点：使用 pipeline 批量写入，减少网络 RTT
                async with self.redis.pipeline() as pipe:
                    pipe.setex(f"{key}:data", ttl, data)
                    pipe.setex(f"{key}:mime", ttl, mime_type)
                    await pipe.execute()
            except Exception: pass

    async def clear(self, pattern: str = "jmcomic:*"):
        # 确保 Redis 已初始化
        await self._ensure_redis()
        
        if self.redis:
            try:
                # 异步扫描和删除
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
            except Exception: pass
        with self.mem_lock:
            self.memory_cache.clear()

class JmClientManager:
    """jmcomic 客户端管理器 (单例复用连接池)"""
    _instance = None
    _client = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self):
        if not self._client:
            with self._lock:
                if not self._client:
                    option = download_manager.get_option()
                    self._client = option.build_jm_client()
        return self._client

class FileService:
    """无状态文件服务：所有方法均需传入 root_dir"""
    
    def _check_safe_path(self, root_dir: Path, rel_path: str) -> Path:
        """防止路径遍历攻击"""
        try:
            clean_rel = os.path.normpath(rel_path).lstrip(os.sep)
            if not clean_rel or clean_rel == '.':
                return root_dir
            full_path = (root_dir / clean_rel).resolve()
            if not str(full_path).startswith(str(root_dir.resolve())):
                raise HTTPException(403, "非法路径访问")
            return full_path
        except Exception:
            raise HTTPException(400, "路径无效")

    def list_files(self, root_dir: Path, rel_path: str, sort_by: str, order: str, page: int, page_size: int):
        target_dir = self._check_safe_path(root_dir, rel_path)
        
        if not target_dir.exists():
            return {"items": [], "total": 0, "page": 1, "total_pages": 0}
            
        items = []
        try:
            with os.scandir(target_dir) as entries:
                for entry in entries:
                    if entry.name.startswith('.'): continue
                    is_dir = entry.is_dir()
                    
                    item = {
                        "name": entry.name,
                        "path": os.path.relpath(entry.path, root_dir).replace('\\', '/'),
                        "type": "directory" if is_dir else "file"
                    }
                    
                    # 性能优化：仅当按时间排序时才获取 stat
                    if sort_by == 'time':
                        stat = entry.stat()
                        item["mtime"] = stat.st_mtime
                        item["size"] = stat.st_size if not is_dir else 0
                    else:
                        item["mtime"] = 0
                        
                    items.append(item)
        except OSError as e:
            raise HTTPException(500, f"读取目录失败: {e}")

        # 排序
        reverse = (order == "desc")
        def natural_key(text):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

        dirs = [x for x in items if x["type"] == "directory"]
        files = [x for x in items if x["type"] == "file"]

        if sort_by == "time":
            dirs.sort(key=lambda x: x["mtime"], reverse=reverse)
            files.sort(key=lambda x: x["mtime"], reverse=reverse)
        else:
            dirs.sort(key=lambda x: natural_key(x["name"]), reverse=reverse)
            files.sort(key=lambda x: natural_key(x["name"]), reverse=reverse)
            
        all_items = dirs + files
        
        # 分页
        total = len(all_items)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            "items": all_items[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total > 0 else 1
        }

    def delete_path(self, root_dir: Path, rel_path: str):
        target = self._check_safe_path(root_dir, rel_path)
        if not target.exists(): raise HTTPException(404, "目标不存在")
        try:
            if target.is_dir(): shutil.rmtree(target)
            else: target.unlink()
        except Exception as e:
            raise HTTPException(500, f"删除失败: {e}")

    def generate_thumbnail(self, root_dir: Path, rel_path: str, width: int, height: int) -> Tuple[bytes, str]:
        """
        生成缩略图
        返回: (图片数据, MIME类型)
        """
        target = self._check_safe_path(root_dir, rel_path)
        img_path = None
        
        # 寻找第一张图片
        if target.is_dir():
            try:
                # 简单排序找第一张
                files = sorted([f for f in os.listdir(target) if f.lower().endswith(('.jpg','.png','.webp','.jpeg'))])
                if files: img_path = target / files[0]
            except: pass
        elif target.is_file():
            img_path = target
            
        if not img_path: raise HTTPException(404, "无图片")
        
        try:
            with Image.open(img_path) as img:
                # 检测源图片格式
                source_format = img.format.lower() if img.format else ''
                source_ext = img_path.suffix.lower()
                
                # 确定输出格式
                thumbnail_format = API_CONFIG["image"]["thumbnail_format"].lower()
                if thumbnail_format == 'auto':
                    # 自动模式：如果源图片是WebP则输出WebP，否则输出JPEG
                    if source_format == 'webp' or source_ext == '.webp':
                        output_format = 'WEBP'
                        mime_type = 'image/webp'
                    else:
                        output_format = 'JPEG'
                        mime_type = 'image/jpeg'
                elif thumbnail_format == 'webp':
                    output_format = 'WEBP'
                    mime_type = 'image/webp'
                else:  # jpeg 或其他
                    output_format = 'JPEG'
                    mime_type = 'image/jpeg'
                
                # 转换为RGB模式（WebP和JPEG都需要RGB）
                if output_format == 'JPEG':
                    img = img.convert('RGB')
                elif output_format == 'WEBP':
                    # WebP支持RGBA，如果原图有透明通道则保留
                    if img.mode in ('RGBA', 'LA'):
                        pass  # 保持RGBA模式
                    else:
                        img = img.convert('RGB')
                
                # 缩放逻辑
                img_ratio = img.width / img.height
                target_ratio = width / height
                if img_ratio > target_ratio:
                    new_height = height
                    new_width = int(height * img_ratio)
                else:
                    new_width = width
                    new_height = int(width / img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                # 中心裁剪
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                img = img.crop((left, top, left + width, top + height))
                
                output = BytesIO()
                # 从配置文件读取缩略图专用压缩质量（比普通图片质量低，提升性能）
                thumbnail_quality = API_CONFIG["image"].get("thumbnail_quality", 75)
                image_quality = API_CONFIG["image"]["image_quality"]
                # 使用缩略图专用质量（如果配置了），否则使用普通质量
                quality = thumbnail_quality if "thumbnail_quality" in API_CONFIG["image"] else image_quality
                
                # 优化：根据格式使用不同的保存参数
                save_kwargs = {"format": output_format, "quality": quality}
                
                # WebP 格式优化：添加 method 参数（0-6，值越大压缩越好但越慢）
                # 对于缩略图，使用 method=4 在速度和压缩率之间取得平衡
                if output_format == 'WEBP':
                    save_kwargs["method"] = 4  # 平衡压缩速度和文件大小
                
                # JPEG 格式优化：使用 optimize=True 可以进一步减小文件大小
                if output_format == 'JPEG':
                    save_kwargs["optimize"] = True
                
                img.save(output, **save_kwargs)
                return output.getvalue(), mime_type
        except Exception as e:
            raise HTTPException(500, f"缩略图错误: {e}")

# ==================== 下载管理器 ====================

class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.lock = threading.Lock()
    
    async def connect(self, ws: WebSocket, album_id: str):
        await ws.accept()
        with self.lock:
            self.connections[album_id].add(ws)
            
    def disconnect(self, ws: WebSocket, album_id: str):
        with self.lock:
            self.connections[album_id].discard(ws)
            
    async def broadcast(self, album_id: str, data: dict):
        with self.lock:
            targets = self.connections.get(album_id, set()).copy()
        
        if not targets: return
        msg = json.dumps({"type": "progress", "album_id": album_id, "progress": data}, ensure_ascii=False)
        for ws in targets:
            try: await ws.send_text(msg)
            except: pass

connection_manager = ConnectionManager()

class DownloadManager:
    def __init__(self):
        self.progress_store = {}
        self.progress_lock = threading.Lock()
        # 从配置文件读取线程池最大工作线程数
        max_workers = API_CONFIG["thread_pool"]["max_workers"]
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.option_file = APP_CONFIG["option_file"]
        self._option = None

    def get_option(self):
        if not self._option:
            # 动态更新 option.yml 路径
            if NAS_CONFIG.get('enabled') and DEFAULT_IMAGE_DIR.exists():
                try:
                    with open(self.option_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                    if 'dir_rule' not in data: data['dir_rule'] = {}
                    
                    # 只有路径不同才写入
                    if data['dir_rule'].get('base_dir') != str(DEFAULT_IMAGE_DIR) + '/':
                        data['dir_rule']['base_dir'] = str(DEFAULT_IMAGE_DIR) + '/'
                        with open(self.option_file, 'w', encoding='utf-8') as f:
                            yaml.dump(data, f, allow_unicode=True)
                except Exception as e:
                    print(f"配置更新警告: {e}")
            self._option = jmcomic.create_option_by_file(self.option_file)
        return self._option

    def _log_callback(self, album_id, topic, msg):
        if 'image.before' in topic or 'image.after' in topic:
            match = re.search(r'\[(\d+)/(\d+)\]', msg)
            if match:
                curr, total = int(match.group(1)), int(match.group(2))
                with self.progress_lock:
                    self.progress_store[album_id] = {
                        "current": curr, "total": total, "status": "downloading",
                        "message": f"下载中 {curr}/{total}",
                        "percentage": int(curr/total*100) if total else 0
                    }
                # 异步推送
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            connection_manager.broadcast(album_id, self.progress_store[album_id]), loop
                        )
                except: pass

    def download_sync(self, album_id: str):
        # 这里的 Hook 依然存在并发覆盖问题，jmcomic 库设计限制
        # 但在低并发下可接受
        def hook(topic, msg): self._log_callback(album_id, topic, msg)
        
        original_log = jmcomic.JmModuleConfig.EXECUTOR_LOG
        jmcomic.JmModuleConfig.EXECUTOR_LOG = hook
        
        try:
            with self.progress_lock:
                self.progress_store[album_id] = {"status": "starting", "message": "初始化..."}
            
            jmcomic.download_album(album_id, self.get_option())
            
            with self.progress_lock:
                self.progress_store[album_id] = {"status": "completed", "message": "完成", "percentage": 100}
            return {"success": True}
        except Exception as e:
            with self.progress_lock:
                self.progress_store[album_id] = {"status": "error", "message": str(e)}
            raise e
        finally:
            jmcomic.JmModuleConfig.EXECUTOR_LOG = original_log

download_manager = DownloadManager()
cache_service = CacheService()
file_service = FileService()
jm_client_manager = JmClientManager()

# ==================== FastAPI App ====================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# 从配置文件读取GZip最小压缩大小
gzip_min_size = API_CONFIG["compression"]["gzip_minimum_size"]
app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)

# 静态文件挂载
if os.path.exists(APP_CONFIG["static_dir"]):
    app.mount("/static", StaticFiles(directory=APP_CONFIG["static_dir"]), name="static")

# ==================== 依赖注入 (核心修正) ====================

def get_current_root_dir(comic_dir: Optional[str] = Query(None)) -> Path:
    """
    Dependency: 根据请求参数动态决定文件操作的根目录
    """
    # 如果没有指定 comic_dir 或者指定为默认的 'JM'，使用默认路径
    if not comic_dir or comic_dir == 'JM':
        return DEFAULT_IMAGE_DIR
    
    # 查找配置中的其他目录
    dirs = NAS_CONFIG.get('comic_dirs', [])
    for d in dirs:
        if d.get('name') == comic_dir:
            path_str = d.get('path', '')
            # 同样需要处理路径拼接问题
            if NAS_CONFIG.get('enabled'):
                nas_mount = Path(NAS_CONFIG['mount_point'])
                # 如果配置里的 path 已经是完整挂载路径（以挂载点开头），直接用
                if path_str.startswith(str(nas_mount)):
                    p = Path(path_str)
                else:
                    # 否则视为相对路径进行拼接
                    p = nas_mount / path_str.lstrip('/')
            else:
                p = Path(path_str)
            if p.exists():
                return p.resolve()
            else:
                print(f"⚠ 警告: 请求的目录不存在: {p}")
            
    return DEFAULT_IMAGE_DIR
# ==================== API 路由 ====================

@app.get("/")
async def root():
    return {"status": "running", "version": "2.0"}

@app.get("/index.html", include_in_schema=False)
@app.get("/viewer.html", include_in_schema=False)
@app.get("/browse.html", include_in_schema=False)
async def serve_pages(request: Any):
    # 简单的页面路由
    path = request.url.path.strip('/')
    f = Path(APP_CONFIG["static_dir"]) / path
    if f.exists(): return FileResponse(f)
    return Response(status_code=404)

@app.get("/api/comic-dirs")
async def get_dirs():
    """获取可用目录列表"""
    dirs = NAS_CONFIG.get('comic_dirs', [])
    # 构造默认 JM
    default = {"name": "JM", "display_name": "JM (默认)", "path": str(DEFAULT_IMAGE_DIR)}
    return {"success": True, "dirs": [default] + dirs, "current": "JM"}

@app.get("/api/files")
async def api_list_files(
    path: str = "",
    sort_by: str = "time",
    order: str = "desc",
    page: int = 1,
    page_size: int = None,
    _t: Optional[int] = None,
    root_dir: Path = Depends(get_current_root_dir),
    comic_dir: str = "JM"
):
    # 从配置文件读取默认分页大小
    if page_size is None:
        page_size = API_CONFIG["pagination"]["default_page_size"]
    # 缓存键包含 comic_dir
    key = cache_service.get_key("files", comic_dir, path, sort_by, order, page, page_size)
    
    if _t is None:
        cached = await cache_service.get(key)
        if cached: return JSONResponse(cached, headers={"X-Cache": "HIT"})

    # 放到线程池执行 IO
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, file_service.list_files, root_dir, path, sort_by, order, page, page_size)
        res = {"success": True, "path": path, **data}
        
        if _t is None: await cache_service.set(key, res)
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/image/{path:path}")
async def api_get_image(path: str, root_dir: Path = Depends(get_current_root_dir)):
    """高性能流式传输图片"""
    try:
        # 使用私有方法检查路径，不修改全局状态
        file_path = file_service._check_safe_path(root_dir, path)
        if not file_path.is_file(): raise HTTPException(404)
        
        # 从配置文件读取图片缓存时间
        cache_max_age = API_CONFIG["image"]["cache_max_age"]
        return FileResponse(file_path, headers={"Cache-Control": f"public, max-age={cache_max_age}"})
    except HTTPException: raise
    except Exception: raise HTTPException(404)

@app.get("/api/thumbnail/{path:path}")
async def api_thumbnail(
    path: str, width: int = None, height: int = None, 
    root_dir: Path = Depends(get_current_root_dir),
    comic_dir: str = "JM"
):
    # 从配置文件读取默认缩略图尺寸
    if width is None:
        width = API_CONFIG["image"]["thumbnail_default_width"]
    if height is None:
        height = API_CONFIG["image"]["thumbnail_default_height"]
    
    key = cache_service.get_key("thumb", comic_dir, path, width, height)
    
    # 1. 异步获取缓存
    cached = await cache_service.get_thumbnail(key)
    if cached:
        cached_data, cached_mime = cached
        thumbnail_cache_age = API_CONFIG["image"]["thumbnail_cache_max_age"]
        return Response(cached_data, media_type=cached_mime, headers={"X-Cache": "HIT", "Cache-Control": f"public, max-age={thumbnail_cache_age}"})

    try:
        loop = asyncio.get_event_loop()
        # 2. 重点：使用 image_executor 独立线程池
        data, mime_type = await loop.run_in_executor(
            image_executor, 
            file_service.generate_thumbnail, 
            root_dir, path, width, height
        )
        
        # 3. 异步写入缓存
        await cache_service.set_thumbnail(key, data, mime_type)
        return Response(data, media_type=mime_type, headers={"X-Cache": "MISS"})
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.delete("/api/files")
async def api_delete(path: str, root_dir: Path = Depends(get_current_root_dir)):
    try:
        file_service.delete_path(root_dir, path)
        await cache_service.clear()
        return {"success": True}
    except Exception as e: raise HTTPException(500, str(e))

# ==================== JMComic 代理接口 ====================

class DownloadReq(BaseModel):
    album_id: str
    option_file: Optional[str] = None

@app.post("/download")
async def api_download(req: DownloadReq):
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(download_manager.executor, download_manager.download_sync, req.album_id)
        return {"success": True, "message": "任务已提交"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.websocket("/ws/progress/{album_id}")
async def ws_progress(ws: WebSocket, album_id: str):
    await connection_manager.connect(ws, album_id)
    try:
        # 发送当前状态
        with download_manager.progress_lock:
            curr = download_manager.progress_store.get(album_id)
        if curr: await ws.send_text(json.dumps({"type":"progress", "progress":curr}))
        
        while True: 
            await ws.receive_text() # 保持连接
    except:
        connection_manager.disconnect(ws, album_id)

# 代理分类与排行 (复用 Client)
class CategoryReq(BaseModel):
    page: int = 1
    time: str = 'a'
    category: str = '0'
    order_by: str = 'mv'

@app.post("/api/category")
async def api_category(req: CategoryReq):
    try:
        # 获取客户端 (这里通常是轻量的，不需要 await)
        client = jm_client_manager.get_client()
        
        # 重点：将同步的 categories_filter 放入线程池执行
        loop = asyncio.get_event_loop()
        page = await loop.run_in_executor(
            None,  # 这里可以用默认线程池，因为它属于 IO 操作
            lambda: client.categories_filter(req.page, req.time, req.category, req.order_by)
        )
        
        albums = []
        # iter_id_title_tag 是内存操作，速度很快，可以直接跑
        for aid, info in page.iter_id_title_tag():
            albums.append({
                "album_id": aid,
                "title": info.get('name'),
                "tags": info.get('tags', []),
                "author": info.get('author'),
                "cover_url": f"https://{client.get_html_domain()}/media/albums/{aid}.jpg"  # 简易拼接
            })
            
        return {"success": True, "albums": albums, "total": len(albums)}
    except Exception as e:
        # 客户端可能过期，重置
        jm_client_manager._client = None
        raise HTTPException(500, str(e))

@app.post("/api/cache/clear")
async def api_clear_cache():
    await cache_service.clear()
    return {"success": True}

health_check = app.get("/health")
async def health_check():
    return {"success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_CONFIG["host"], port=SERVER_CONFIG["port"])