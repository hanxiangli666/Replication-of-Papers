import sys
import subprocess
import importlib.util

def install_package(package_name):
    """自动安装缺失的库"""
    print(f"正在安装 {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"✅ {package_name} 安装完成。")

def check_cuda():
    """检查显卡和 CUDA 状态"""
    print("\n--- 1. 检查 PyTorch 和 CUDA ---")
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ 发现 {device_count} 个 CUDA 设备:")
            for i in range(device_count):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # 简单的显存估算提醒
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)
            print(f"   - 显存总量: {total_memory:.2f} GB")
            
            if total_memory < 16:
                print("⚠️ 提示: 您的显存小于 16GB。")
                print("   建议后续加载模型时使用 'load_in_4bit=True' (4-bit量化) 以防显存不足。")
            else:
                print("✅ 显存充足，可以尝试运行半精度 (FP16) 模型。")
        else:
            print("❌ 未检测到 CUDA 设备！请确认您安装了 PyTorch CUDA 版本。")
            print("   如果这一步失败，后续代码将无法使用显卡加速。")
    except ImportError:
        print("❌ 未检测到 PyTorch。请先根据 pytorch.org 的指引安装 PyTorch。")

def install_dependencies():
    """安装 CAG 项目和 Llama 3.1 所需的依赖"""
    print("\n--- 2. 安装依赖库 ---")
    # 核心依赖列表
    # transformers: 加载模型
    # accelerate: 帮助模型在 GPU 上高效运行
    # bitsandbytes: 用于 4-bit 量化（这对个人电脑运行 Llama 3.1 至关重要）
    # protobuf, scipy: 辅助数学库
    requirements = [
        "transformers>=4.43.0", 
        "accelerate>=0.26.0", 
        "bitsandbytes", 
        "protobuf",
        "scipy"
    ]
    
    for lib in requirements:
        # 提取包名进行检查
        pkg_name = lib.split(">")[0].split("=")[0]
        if importlib.util.find_spec(pkg_name) is None:
            install_package(lib)
        else:
            print(f"✅ {pkg_name} 已安装")

if __name__ == "__main__":
    print("🚀 开始环境配置检查...\n")
    
    # 1. 检查 CUDA
    check_cuda()
    
    # 2. 安装缺失的依赖
    install_dependencies()
    
    print("\n🎉 环境配置检查结束！")
    print("下一步：您可以在 VS Code 中打开刚才解压的 CAG 文件夹，尝试运行代码了。")