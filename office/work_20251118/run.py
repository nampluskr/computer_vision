import subprocess
import sys

def run(file_list):

    for i, file in enumerate(file_list, 1):
        print(f"\n[{i}/{len(file_list)}] Running: {file}")
        try:
            result = subprocess.run([sys.executable, file], check=True,  encoding="utf-8")
            print(f"Completed: {file}")
        except subprocess.CalledProcessError:
            print(f"Error: Script '{file}' failed during execution")
            break  
        except FileNotFoundError:
            print(f"Error: Script '{file}' not found")
            break

if __name__ == "__main__":

    file_list = [
        # "01_dcgan_bce.py", 
        # "02_dcgan_hinge.py", 
        "03_dcgan_wgp.py",
        # "04_resgan_bce.py",
        # "05_sn-resgan_bce.py",
        # "06_resgan_hinge.py",
        # "07_sn-resgan_hinge.py",
    ]

    run(file_list)
