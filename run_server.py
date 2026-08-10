"""启动脚本 - 使用 assist conda 环境"""
import subprocess
import sys
import os

# 切换到项目目录
project_dir = r"D:\knowledge\project\ai-super-tutor"
os.chdir(project_dir)

# 使用 assist 环境的 python 启动
python_exe = r"D:\program\Anaconda\envs\assist\python.exe"
script = os.path.join(project_dir, "agent.py")

print(f"Starting server from: {project_dir}")
print(f"Python: {python_exe}")
print(f"Script: {script}")

result = subprocess.run([python_exe, script], cwd=project_dir)
