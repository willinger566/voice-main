# 运行环境在哪里，能否删除

## 当前实际使用的环境

完整目录：`D:\s3\s4\voice\voice-main\.venv-ser`，约 **5.25 GiB**。

启动入口：`.venv-ser\Scripts\python.exe`。名字里的ser来自语音情绪识别，虽然最初用于训练，现在同时是网页的正式运行环境。里面的PyTorch/CUDA、FunASR、NumPy、openSMILE、FastAPI等依赖都是网页推理所需，不能把整个目录当训练缓存。

它由`python -m venv`创建，不是Conda环境，通常不会列在`conda env list`中。启动脚本直接调用其中的解释器，不需要先执行`conda activate`。

本项目训练没有使用base环境。即使终端提示符显示`(base)`，`start.ps1`仍显式调用上面的解释器入口。

## 与Conda环境的区别

| 环境 | 类型与用途 | 和本项目的关系 | 现在能否直接删除 |
|---|---|---|---|
| `.venv-ser` | Python venv，安装包位于项目目录 | 正在运行网页与模型 | **不能直接删**；删后须先重建依赖或改用另一个配置好的环境 |
| `D:\Anaconda\envs\cvhw` | 你已有的Conda环境，包含Python解释器、标准库及自己的包 | 提供创建venv所用的Python3.11.14基础运行时 | **目前不能删**，否则本项目venv也可能无法启动 |
| `D:\Anaconda\envs\slicer` | 另一个Conda环境 | 当前项目不依赖；此前检查其NumPy/PyTorch导入不完整，故没有使用或修改它 | 就本项目而言不需要；确认其他项目不用后可自行删除 |
| 项目旧`.venv` | 早期辅助venv，继承cvhw的site-packages | 已经不再使用 | **本轮已删除** |
| Anaconda的base | Conda基础环境 | 没有用来训练或作为当前网页的Python环境 | 本轮未修改它 |

**“独立安装包”不等于“完全自带Python”。**

`.venv-ser/pyvenv.cfg`设置`include-system-site-packages = false`，用户site也关闭。因此本项目的torch、numpy等包从`.venv-ser/Lib/site-packages`加载，不继承cvhw或slicer里的包。修改cvhw的pip包通常不会改变本项目的对应包。

但`home`和`sys.base_prefix`仍指向`D:\Anaconda\envs\cvhw`，标准库例如`os.py`也来自该处。venv依赖这个Python基础运行时，而非一个完整、可随意移动的独立Conda环境。因此不能先删除cvhw，再期待venv继续工作。

可运行以下命令检查实际来源，已保存的结果位于`results/runtime_environment.json`：

```powershell
.\.venv-ser\Scripts\python.exe scripts/inspect_environment.py
```

## 如果要删除或更换它

可以更换环境，**不需要重新训练模型**。正确顺序是：停止服务，用另一个完整的Python3.11安装创建新的venv，在新环境安装匹配的PyTorch/torchaudio及`requirements.txt`，修改启动入口并通过`verify_runtime.py`后，再删除旧环境。

示例仅展示重建方式，不会替你删除现有目录：

```powershell
# 选择你确认可用、且打算保留的Python3.11；新目录避免覆盖当前运行环境
D:\Anaconda\envs\slicer\python.exe -m venv .venv-new
# CUDA12.8官方安装示例；CPU机器应选择CPU构建
.\.venv-new\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-new\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-new\Scripts\python.exe -m pip check
.\.venv-new\Scripts\python.exe run.py
```

这里即使slicer原来的NumPy损坏，只要它的Python解释器完整，创建的新venv也不会继承损坏的site-packages；若解释器本身也有问题，应换另一个完整的Python安装。新venv仍依赖你选择的基础Python，不能连同它一起删除。PyTorch安装参考[官方说明](https://pytorch.org/get-started/previous-versions/)。本机原实际版本为torch2.11开发版+CUDA12.8，精确清单见`results/runtime_environment.lock.txt`。

删除`.venv-ser`本身不会删除`models/production`中的训练结果，但会失去运行依赖，使当前`start.ps1`无法启动。切换到`.venv-new`后也需要修改`start.ps1`里的解释器路径，或者直接使用新解释器执行`run.py`。

本轮没有删除、卸载或重建你的slicer、cvhw、base环境；仅删除项目旧`.venv`及当前venv里不再使用的XGBoost。pyarrow仍被openSMILE的依赖链使用，已保留并通过pip依赖检查。
