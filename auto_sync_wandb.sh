#!/bin/bash

# ===================== 配置项（已按你的需求修改） =====================
# wandb离线日志的根目录
WANDB_LOG_ROOT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/tab_model/Orion-MSP-main/wandb/dir/wandb"
# 脚本运行日志保存路径（当前目录）
SYNC_LOG_FILE="./wandb_auto_sync.log"
# 同步间隔（1小时=3600秒）
SYNC_INTERVAL=3600
# 你的wandb API Key（必填，用于每次同步前登录）
WANDB_API_KEY="2f4b01f1da6026ab2405638e7b9f3c4406ef50e1"

# ===================== 初始化操作 =====================
echo -e "\n=====================================" >> $SYNC_LOG_FILE
echo "[$(date +'%Y-%m-%d %H:%M:%S')] wandb自动同步脚本启动" >> $SYNC_LOG_FILE
echo "同步根目录: $WANDB_LOG_ROOT" >> $SYNC_LOG_FILE
echo "同步间隔: $SYNC_INTERVAL 秒" >> $SYNC_LOG_FILE
echo "脚本日志保存路径: $SYNC_LOG_FILE" >> $SYNC_LOG_FILE

# ===================== 核心同步逻辑 =====================
while true; do
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] 开始新一轮wandb日志同步" >> $SYNC_LOG_FILE

    # 步骤1：每次同步前强制重新登录wandb
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 执行wandb登录（强制重新登录）" >> $SYNC_LOG_FILE
    wandb login --relogin $WANDB_API_KEY >> $SYNC_LOG_FILE 2>&1

    # 检查登录是否成功
    if [ $? -ne 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 错误：wandb登录失败！本轮同步终止" >> $SYNC_LOG_FILE
        sleep $SYNC_INTERVAL
        continue
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] wandb登录成功" >> $SYNC_LOG_FILE

    # 步骤2：检查日志根目录是否存在
    if [ ! -d "$WANDB_LOG_ROOT" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 错误：日志根目录 $WANDB_LOG_ROOT 不存在！" >> $SYNC_LOG_FILE
        sleep $SYNC_INTERVAL
        continue
    fi

    # 步骤3：遍历并同步所有offline-run目录
    for run_dir in "$WANDB_LOG_ROOT"/offline-run-*; do
        if [ ! -d "$run_dir" ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 未检测到新的offline-run目录，跳过" >> $SYNC_LOG_FILE
            continue
        fi

        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始同步目录: $run_dir" >> $SYNC_LOG_FILE
        wandb sync "$run_dir" >> $SYNC_LOG_FILE 2>&1

        if [ $? -eq 0 ]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 目录 $run_dir 同步完成" >> $SYNC_LOG_FILE
        else
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 错误：目录 $run_dir 同步失败！" >> $SYNC_LOG_FILE
        fi
    done

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 本轮同步完成，将休眠 $SYNC_INTERVAL 秒" >> $SYNC_LOG_FILE
    sleep $SYNC_INTERVAL
done