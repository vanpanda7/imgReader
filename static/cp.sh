#!/bin/bash

# 部署脚本：将静态文件复制到 Web 服务器目录
# 使用方法：在 static 目录下运行 ./cp.sh

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 源文件目录（脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"

# 目标目录
TARGET_DIR="/opt/1panel/www/sites/rick.monster/index"

# 检查目标目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}错误: 目标目录不存在: $TARGET_DIR${NC}"
    exit 1
fi

# 要复制的文件列表
FILES=(
    "viewer.html"
    "index.html"
)

# 复制文件
echo -e "${YELLOW}开始复制文件...${NC}"
SUCCESS=true

for file in "${FILES[@]}"; do
    SOURCE_FILE="$SOURCE_DIR/$file"
    TARGET_FILE="$TARGET_DIR/$file"
    
    if [ ! -f "$SOURCE_FILE" ]; then
        echo -e "${RED}错误: 源文件不存在: $SOURCE_FILE${NC}"
        SUCCESS=false
        continue
    fi
    
    # 先删除目标文件（如果存在）
    if [ -f "$TARGET_FILE" ]; then
        if rm "$TARGET_FILE"; then
            echo -e "${YELLOW}→${NC} 已删除旧文件: $file"
        else
            echo -e "${RED}警告: 删除旧文件失败: $file${NC}"
            # 继续尝试复制，不中断流程
        fi
    fi
    
    # 复制文件
    if cp "$SOURCE_FILE" "$TARGET_FILE"; then
        echo -e "${GREEN}✓${NC} 已复制: $file"
    else
        echo -e "${RED}✗${NC} 复制失败: $file"
        SUCCESS=false
    fi
done

# 输出结果
if [ "$SUCCESS" = true ]; then
    echo -e "\n${GREEN}所有文件复制成功！${NC}"
    exit 0
else
    echo -e "\n${RED}部分文件复制失败，请检查错误信息${NC}"
    exit 1
fi
