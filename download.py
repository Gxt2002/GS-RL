import os
from huggingface_hub import snapshot_download

def download_hf_folder():
    # 设置仓库 ID 和目标下载子目录
    repo_id = "ydcttt/ReconDreamer-RL"
    allow_patterns = "assets/nus/*"
    
    # 设置本地保存路径（默认当前目录下的 local_assets）
    local_dir = "/mnt/data/users/perception-users/gongxintian/ReconDreamer-RL"
    
    print(f"开始从仓库 {repo_id} 下载 {allow_patterns} ...")
    
    try:
        # 使用 snapshot_download 下载指定匹配模式的文件
        local_folder = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",          # 明确指定是 dataset 类型的仓库
            allow_patterns=allow_patterns, # 只允许下载该路径匹配的文件
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 直接复制文件而非创建软链接，方便后续使用
            resume_download=True           # 开启断点续传
        )
        print(f"下载成功！文件已保存至: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"下载过程中出现错误: {e}")

if __name__ == "__main__":
    download_hf_folder()