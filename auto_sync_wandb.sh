#!/bin/bash

# ===================== 配置项（已按你的需求修改） =====================
# wandb离线日志的根目录（你的实际离线日志上级目录）
WANDB_LOG_ROOT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main/wandb/dir/wandb"
# 脚本运行日志保存路径（当前目录下的wandb_auto_sync.log）
SYNC_LOG_FILE="./wandb_auto_sync.log"
# 同步间隔（单位：秒，1小时=3600秒）
SYNC_INTERVAL=3600

# ===================== 初始化操作 =====================
# 写入脚本启动日志
echo -e "\n=====================================" >> $SYNC_LOG_FILE
echo "[$(date +'%Y-%m-%d %H:%M:%S')] wandb自动同步脚本启动" >> $SYNC_LOG_FILE
echo "同步根目录: $WANDB_LOG_ROOT" >> $SYNC_LOG_FILE
echo "同步间隔: $SYNC_INTERVAL 秒" >> $SYNC_LOG_FILE
echo "脚本日志保存路径: $SYNC_LOG_FILE" >> $SYNC_LOG_FILE

# ===================== 核心同步逻辑 =====================
# 无限循环执行同步
while true; do
    # 记录本轮同步开始时间
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] 开始新一轮wandb日志同步" >> $SYNC_LOG_FILE

    # 检查日志根目录是否存在
    if [ ! -d "$WANDB_LOG_ROOT" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 错误：日志根目录 $WANDB_LOG_ROOT 不存在！" >> $SYNC_LOG_FILE
        sleep $SYNC_INTERVAL
        continue
    fi

    # 遍历所有以offline-run-开头的目录（你的实际离线日志目录特征）
    for run_dir in "$WANDB_LOG_ROOT"/offline-run-*; do
        # 跳过非目录文件（避免匹配不到时处理空值）
        if [ ! -d "$run_dir" ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 未检测到新的offline-run目录，跳过" >> $SYNC_LOG_FILE
            continue
        fi

        # 执行同步并将输出写入日志
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始同步目录: $run_dir" >> $SYNC_LOG_FILE
        wandb sync "$run_dir" >> $SYNC_LOG_FILE 2>&1

        # 检查同步是否成功
        if [ $? -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 目录 $run_dir 同步完成" >> $SYNC_LOG_FILE
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 错误：目录 $run_dir 同步失败！" >> $SYNC_LOG_FILE
        fi
    done

    # 记录本轮同步结束时间，然后休眠指定时长
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 本轮同步完成，将休眠 $SYNC_INTERVAL 秒" >> $SYNC_LOG_FILE
    sleep $SYNC_INTERVAL
done