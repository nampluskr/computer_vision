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
        # "01_dcgan_bce.py", 
        # "02_dcgan_mse.py",
        # "03_dcgan_hinge.py", 
        # "04_dcgan_wgp.py",
        "04_dcgan_wgp_relu.py",
        # "04_resgan_bce.py",
        # "05_sn-resgan_bce.py",
        # "06_resgan_hinge.py",
        # "07_sn-resgan_hinge.py",
    ]

    run(file_list)

# >> [1/1] Running: 04_dcgan_wgp_relu.py

# [  1/5] d_loss:-15.399, real_loss:-2.541, fake_loss:-12.858, g_loss:13.173, gp:0.273
# [  2/5] d_loss:-7.456, real_loss:-2.606, fake_loss:-4.850, g_loss:5.091, gp:0.168
# [  3/5] d_loss:-6.573, real_loss:-1.480, fake_loss:-5.092, g_loss:5.139, gp:0.094
# [  4/5] d_loss:-43.401, real_loss:-36.333, fake_loss:-7.068, g_loss:7.150, gp:1.206
# [  5/5] d_loss:-5.856, real_loss:2.679, fake_loss:-8.535, g_loss:8.607, gp:0.193
# >> ./outputs/04_dcgan_wgp_relu-epoch5.png is saved.

# [  1/5] d_loss:-6.919, real_loss:-0.680, fake_loss:-6.239, g_loss:6.287, gp:0.267
# [  2/5] d_loss:-7.372, real_loss:-2.642, fake_loss:-4.731, g_loss:4.759, gp:0.398
# [  3/5] d_loss:-7.129, real_loss:-1.990, fake_loss:-5.139, g_loss:5.217, gp:0.591
# [  4/5] d_loss:-7.021, real_loss:0.184, fake_loss:-7.205, g_loss:7.191, gp:0.411
# [  5/5] d_loss:-7.553, real_loss:1.269, fake_loss:-8.822, g_loss:8.906, gp:0.549
# >> ./outputs/04_dcgan_wgp_relu-epoch10.png is saved.