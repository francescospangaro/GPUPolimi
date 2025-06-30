# GPU CUDA Exercises

Lab exercises for the "GPU & Heterogeneous systems" course 2023/2024 @ Polimi. \
Prof: Antonio Miele Rosario.

This repository contains several CUDA exercise proposed by the professor during the course, you can check them by browsing the branches.
At the moment there are 3 branches: 

---
![Static Badge](https://img.shields.io/badge/1-%7F%20%20%20%20%7F%20%20%20%20%7FVSUM%7F%20%20%20%7F%20%20%20%20%7F-rgb(39%2C%20210%2C%20255)) &ensp;&ensp; Accelerate the computation of the sum of 2 vectors, check for eventual raised error by CUDA commands, compare the time \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;execution results between CPU and GPU.

---
![Static Badge](https://img.shields.io/badge/2%20-%7F%20%20%20%20%7F%20%20%20%20%7FBLUR%7F%20%20%20%20%7F%20%20%20%20%7F-%20rgb(39%2C%20210%2C%20255)) &ensp;&ensp; Accelerate the blurring of a ppm image through the GPU.

---
![Static Badge](https://img.shields.io/badge/3%20-VARIOUS_TESTS-%20rgb(39%2C%20210%2C%20255)) &ensp;&ensp; Examples made available by the professor in order to play around with Nvidia Nsight Compute Profiler (ncu)


# My dev environment

As my development environment i used Visual Studio Code installed on my personal Windows10 machine containing a wsl2 installation.

Specifications:
- Windows10: Version 22H2 (Build SO 19045.4170): It's necessary a Windows10's build version 19044+ with NVIDIA driver r545+ in order to access all CUDA tools(see [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#:~:text=Developer%20tools%20%2D%20Profilers%20%2D%20Volta%20and%20later%20(Using%20Windows%2010%20OS%20build%2019044%2B%20with%20driver%20r545%2B%20or%20using%20Windows%2011%20with%20driver%20r525%2B%20)) for more details)
- GPU Driver: 551.76
- Wsl2: Ubuntu22.04

# CUDA-Toolkit installation

A brief description of the steps that i took in order to make all work properly:
- I had already a wsl2 Ubuntu20.04 distro installed on my machine, but if you haven't just follow the simple NVIDIA "getting started" [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2).

> [!IMPORTANT]
> Ensure that you have the latest WSL kernel installed if you have already a WSL's distro installed.

- Verify if you have gcc installed in your WSL distro by using `gcc --version`. If you don't, run `sudo apt update` and then install it by using `sudo apt install build-essential`.
- At this point if you have a CUDA enabled GPU by running `nvidia-smi` in the WSL's console you should see your GPU informations.
- Follow the installation procedure stated on official CUDA-Toolkit download [page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) after selecting Linux -> x86_64 -> WSL-Ubuntu -> 2.0 -> deb(local).

> [!IMPORTANT]
> At this point you should be able to use CUDA toolkit compiler and profiler, however if like me nothing works try the subsequent steps

- Add at the end of the `.bashrc` file (that can be find at `/home/<yourUsername>`) the following lines: 
    - `export PATH="/usr/local/cuda-12.4/bin:$PATH"`
    - `export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"`
- Refresh the `.bashrc` file using `source .bashrc`
- Check whether cuda-toolkit was successfully installed by using:
    - `nvcc --version` : to see whether the compiler has been correctly installed or not
    - `ncu --version` : to see whether the profiler has been correctly installed or not

# TROUBLESHOOTING

Some problem that i faced and their solutions:
- While trying to use cuda-gdb i faced the error: `Error: get_elf_image(0): Failed to read the ELF image handle 93825002783360 relocated 1, error=CUDBG_ERROR_INVALID_ARGS, error message=`. \
I solved this problem by creating a registry key on Windows as stated [here](https://forums.developer.nvidia.com/t/cuda-gdb-report-internal-error-while-using-under-wsl2/249595).
- While trying to open nsight-systems' GUI through command line i first faced the error `qt.qpa.plugin: Could not load the Qt platform plugin "wayland" in "" even though it was found.` and i solved it by installing qtwayland5 using `sudo apt install qtwayland5`. \
However immediately after I faced another problem: After calling `nsys-ui`, as first thing it says that `OpenGL version is too low (0). Falling back to Mesa software rendering.`, even though i had already OpenGL 4.6 installed on Windows and I subsequently installed it with `sudo add-apt-repository ppa:kisak/kisak-mesa` in wsl too but with no results.
And finally even with that error nsys seems to start but crashes immediately, I don't know if it is related with the OpenGL problem.
The only workaround that i found has been to install nsys on Windows and open the created reports (the CLI commands of nsys works properly) with it.







