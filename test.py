import torch

def check_gpu_support():
    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_gpu_support()