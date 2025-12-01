# 设置指南 (Setup Guide)

## 选项 1: 使用 Python (推荐)

### Windows 安装步骤:

1. **下载并安装 Python**
   - 访问 https://www.python.org/downloads/
   - 下载 Python 3.8 或更高版本
   - **重要**: 安装时勾选 "Add Python to PATH"

2. **验证安装**
   打开新的 PowerShell 或命令提示符，运行:
   ```powershell
   python --version
   ```

3. **安装依赖**
   ```powershell
   pip install numpy scipy
   ```

4. **运行项目**
   ```powershell
   python main.py --matches 10 --rollouts 20
   ```

## 选项 2: 使用 Anaconda/Miniconda

### 安装 Anaconda:

1. **下载 Anaconda**
   - 访问 https://www.anaconda.com/products/distribution
   - 下载 Windows 版本并安装
   - **重要**: 安装时勾选 "Add Anaconda to PATH"

2. **打开 Anaconda Prompt** (不是普通 PowerShell)
   - 在开始菜单搜索 "Anaconda Prompt"

3. **创建环境并安装依赖**
   ```bash
   conda create -n poker python=3.9
   conda activate poker
   pip install numpy scipy
   ```

4. **运行项目**
   ```bash
   python main.py --matches 10 --rollouts 20
   ```

## 选项 3: 使用 Visual Studio Code

如果你使用 VS Code:

1. 安装 Python 扩展
2. 打开项目文件夹
3. VS Code 会自动检测 Python
4. 在终端中运行:
   ```powershell
   pip install numpy scipy
   python main.py
   ```

## 快速测试

安装完成后，运行测试脚本验证安装:
```powershell
python test_basic.py
```

## 常见问题

### "python 不是内部或外部命令"
- Python 未添加到 PATH
- 解决方案: 重新安装 Python 并勾选 "Add to PATH"
- 或手动添加 Python 安装目录到系统 PATH

### "conda 无法识别"
- 使用 Anaconda Prompt 而不是普通 PowerShell
- 或确保 Anaconda 已添加到 PATH

### 依赖安装失败
- 尝试使用国内镜像:
  ```powershell
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy
  ```

## 最小运行示例

如果只想快速测试，可以运行:
```powershell
python main.py --matches 5 --rollouts 10 --opponent mixed
```

这会运行 5 场比赛，每场 20 轮，使用较少的蒙特卡洛模拟（更快但精度较低）。

