# NCNN Infer Template Demo:o:

This project base on [NanoDet image inference](https://github.com/RangiLyu/nanodet/tree/main/demo_ncnn) and [Tencent's NCNN framework](https://github.com/Tencent/ncnn).

# How to build

## Windows
### Step1.
Download and Install Visual Studio from https://visualstudio.microsoft.com/vs/community/

### Step2.
Download and install OpenCV from https://github.com/opencv/opencv/releases

### Step3(Optional).
Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step4.
Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```
Build NCNN following this tutorial: [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)

### Step5.

Add `ncnn_DIR` = `YOUR_NCNN_PATH/build/install/lib/cmake/ncnn` to system environment variables.

Build project: Open x64 Native Tools Command Prompt for VS 2019 or 2017

``` cmd
cd <this-folder>
mkdir -p build
cd build
cmake ..
msbuild nanodet_demo.vcxproj /p:configuration=release /p:platform=x64
```

## Linux

### Step1.
Build and install OpenCV from https://github.com/opencv/opencv

### Step2(Optional).
Download Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step3.
Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git 
```

Build NCNN following this tutorial: [Build for Linux / NVIDIA Jetson / Raspberry Pi](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

### Step4.

Set environment variables. Run:

``` shell script
export ncnn_DIR=YOUR_NCNN_PATH/build/install/lib/cmake/ncnn
```

Build project

``` shell script
cd <this-folder>
mkdir build
cd build
cmake ..
make
```

# Run demo

Download ncnn model such as NanoDet ncnn model.
* [NanoDet ncnn model download link](https://github.com/RangiLyu/nanodet/releases/download/v0.3.0/nanodet_m_ncnn_model.zip)

Copy nanodet_m.param and nanodet_m.bin to demo program folder.

## Webcam

```shell script
nanodet_demo 0 0
```

## Inference images

```shell script
nanodet_demo 1 IMAGE_FOLDER/*.jpg
```

## Inference video

```shell script
nanodet_demo 2 VIDEO_PATH
```

## Benchmark

```shell script
nanodet_demo 3 0
```
![bench_mark](https://github.com/RangiLyu/nanodet/blob/main/demo_ncnn/benchmark.jpg)
****

Notice:

If benchmark speed is slow, try to limit omp thread num.

Linux:

```shell script
export OMP_THREAD_LIMIT=4
```
