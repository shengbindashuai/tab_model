import os
import json
import shutil
import argparse
import subprocess

def find_offline_runs_by_id(target_id, wandb_root="wandb"):
    """
    查找所有包含目标ID的离线运行目录
    :param target_id: 需要合并的wandb运行ID
    :param wandb_root: wandb数据根目录
    :return: 所有匹配的目录路径列表
    """
    run_dirs = []
    for entry in os.listdir(wandb_root):
        entry_path = os.path.join(wandb_root, entry)
        if os.path.isdir(entry_path) and target_id in entry:
            run_dirs.append(entry_path)
    return sorted(run_dirs)  # 按时间排序，确保最早的作为主目录

def merge_history_files(main_dir, other_dirs):
    """
    合并所有其他目录的wandb-history.jsonl到主目录
    :param main_dir: 主目录，作为合并的目标
    :param other_dirs: 其他需要合并的目录
    """
    main_history_path = os.path.join(main_dir, "wandb-history.jsonl")
    for dir_path in other_dirs:
        history_path = os.path.join(dir_path, "wandb-history.jsonl")
        if os.path.exists(history_path):
            with open(history_path, "r") as src, open(main_history_path, "a") as dst:
                shutil.copyfileobj(src, dst)
            print(f"✅ 合并历史记录: {dir_path} → {main_dir}")

def merge_summary_files(main_dir, other_dirs):
    """
    合并所有其他目录的wandb-summary.json到主目录（保留最新值）
    :param main_dir: 主目录，作为合并的目标
    :param other_dirs: 其他需要合并的目录
    """
    main_summary_path = os.path.join(main_dir, "wandb-summary.json")
    main_summary = {}

    # 读取主目录的原始汇总
    if os.path.exists(main_summary_path):
        with open(main_summary_path, "r") as f:
            main_summary = json.load(f)

    # 合并其他目录的汇总（后面的会覆盖前面的，保留最新值）
    for dir_path in other_dirs:
        summary_path = os.path.join(dir_path, "wandb-summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                other_summary = json.load(f)
                main_summary.update(other_summary)
            print(f"✅ 合并汇总数据: {dir_path} → {main_dir}")

    # 写回合并后的汇总
    with open(main_summary_path, "w") as f:
        json.dump(main_summary, f, indent=2)

def sync_to_cloud(main_dir):
    """
    将合并后的主目录同步到wandb云端
    :param main_dir: 合并后的主目录
    """
    try:
        print(f"🔄 开始同步到wandb云端: {main_dir}")
        subprocess.run(["wandb", "sync", main_dir], check=True)
        print("✅ 同步完成！请在wandb官网查看合并后的运行记录。")
    except subprocess.CalledProcessError as e:
        print(f"❌ 同步失败: {e}")
    except FileNotFoundError:
        print("❌ 未找到wandb命令，请确保wandb已正确安装。")

def main():
    parser = argparse.ArgumentParser(description="自动合并并同步相同ID的wandb离线运行记录")
    parser.add_argument("--id", required=True, help="需要合并的wandb运行ID（如 inlfyc74）")
    parser.add_argument("--wandb-root", default="wandb", help="wandb数据根目录（默认: ./wandb）")
    parser.add_argument("--sync", action="store_true", help="合并后是否自动同步到云端")
    args = parser.parse_args()

    # 1. 查找所有同ID的离线目录
    run_dirs = find_offline_runs_by_id(args.id, args.wandb_root)
    if not run_dirs:
        print(f"❌ 未找到ID为 {args.id} 的离线运行目录")
        return
    if len(run_dirs) == 1:
        print(f"⚠️  仅找到1个ID为 {args.id} 的目录，无需合并")
        return

    # 2. 选择最早的目录作为主目录
    main_dir = run_dirs[0]
    other_dirs = run_dirs[1:]
    print(f"📌 选择主目录: {main_dir}")
    print(f"🔍 待合并的其他目录: {other_dirs}")

    # 3. 合并历史记录和汇总数据
    merge_history_files(main_dir, other_dirs)
    merge_summary_files(main_dir, other_dirs)

    # 4. 可选：同步到云端
    if args.sync:
        sync_to_cloud(main_dir)

    print("\n🎉 所有合并操作已完成！")
    print(f"📂 合并后的数据目录: {main_dir}")

if __name__ == "__main__":
    main()