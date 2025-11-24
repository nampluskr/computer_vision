import subprocess
import sys

def run(file_list):

    for i, file in enumerate(file_list, 1):
        print(f"\n>> [{i}/{len(file_list)}] Running: {file}\n")
        try:
            result = subprocess.run([sys.executable, file], check=True,  encoding="utf-8")
            print(f">> Completed: {file}")
        except subprocess.CalledProcessError:
            print(f"[Error] Script '{file}' failed during execution")
            break
        except FileNotFoundError:
            print(f"[Error] Script '{file}' not found")
            break

if __name__ == "__main__":

    file_list = [
        "train_gan_mnist.py",
        "train_gan_fashion.py",
        "train_gan_cifar10.py",
    ]

    run(file_list)
