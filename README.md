# RKNNtest

`RKNNtest` 是一个面向 Rockchip RKNN Runtime 的 C++ 推理测试工程。它用于加载 RKNN 模型，批量读取图片，通过 RGA 做 letterbox 缩放预处理，调用 RKNN Runtime 推理，并把 YOLO 风格检测框绘制到结果图片中。

当前工程默认模型名为 `climb0325_2.rv1126.rknn`，默认输入目录为 `./images`，默认输出目录为 `./results`。

## 功能概览

- 加载 `.rknn` 模型并查询输入、输出 tensor 信息。
- 遍历图片目录，支持 `.bmp`、`.jpg`、`.jpeg`、`.png`，大小写均可。
- 使用 `stb_image` 读取图片，使用 RGA 将图片 letterbox 到模型输入尺寸。
- 自动根据模型输入 tensor 推断输入宽高和通道数。
- 自动根据 3 个输出 tensor 推断 YOLO grid 尺寸和类别数。
- 使用固定 YOLO anchors 完成解码、置信度过滤和 NMS。
- 将检测框绘制回原图，并保存为 BMP 结果图。
- 提供 `gen_test_img` 工具生成 3 张基础测试图片。

## 目录结构

```text
RKNNtest/
|-- CMakeLists.txt              # CMake 构建入口
|-- test.cpp                    # RKNN + RGA 批量推理主程序
|-- gen_test_img.cpp            # 测试图片生成工具
|-- images/                     # 默认输入图片目录
|-- results/                    # 默认输出目录，运行时自动创建
|-- libs/aarch64/               # RKNN / RGA aarch64 头文件和动态库
|   |-- include/
|   `-- lib/
|-- compat/                     # PC mock / Windows 兼容头文件
|-- stb_image.h                 # 图片读取单头文件库
`-- stb_image_write.h           # 图片写入单头文件库
```

## 运行环境

推荐在 Rockchip aarch64 Linux 设备上构建和运行，例如 RV1126/RV1109/RK 系列设备，具体取决于你的 RKNN 模型和 Runtime 版本。

基础依赖：

- CMake 3.14 或更高版本
- 支持 C++17 的编译器
- Rockchip RKNN Runtime
- Rockchip RGA Runtime
- 项目内置的 `libs/aarch64/include` 和 `libs/aarch64/lib`
- 项目内置的 `libs/aarch64/include/rga` 和 `libs/aarch64/lib/librga.so`

当前 `CMakeLists.txt` 会链接以下库：

```text
libs/aarch64/lib/librknnrt.so
libs/aarch64/lib/librga.so
```

`gen_test_img` 只生成 BMP 测试图，不链接 RKNN/RGA 动态库。

运行前通常需要把项目内置动态库加入运行时搜索路径：

```bash
export LD_LIBRARY_PATH=$PWD/libs/aarch64/lib:$LD_LIBRARY_PATH
```

## 模型文件

默认模型路径写在 `test.cpp` 中：

```cpp
#define DEFAULT_MODEL_PATH  "climb0325_2.rv1126.rknn"
```

该模型文件没有提交到仓库，并且已在 `.gitignore` 中忽略。使用默认命令运行前，需要把模型放到项目根目录：

```text
RKNNtest/climb0325_2.rv1126.rknn
```

也可以在运行时显式传入模型路径：

```bash
./build/test /path/to/model.rknn ./images ./results
```

## 构建

建议先清理旧的 `build` 目录再重新生成。当前仓库里的 `build/` 可能是历史构建产物，不一定对应当前源码和本机路径。

```bash
cd RKNNtest
rm -rf build
cmake -S . -B build
cmake --build build -j
```

构建完成后会生成两个程序：

```text
build/test
build/gen_test_img
```

在 Windows/MinGW 环境下文件名可能带 `.exe` 后缀。

## 生成测试图片

项目已经带有 `images/test1.bmp`、`images/test2.bmp`、`images/test3.bmp`。如果需要重新生成，可以运行：

```bash
./build/gen_test_img
```

生成内容：

- `test1.bmp`：白底红色矩形
- `test2.bmp`：黑底绿色圆形区域
- `test3.bmp`：蓝绿色渐变图

## 运行推理

使用默认参数：

```bash
export LD_LIBRARY_PATH=$PWD/libs/aarch64/lib:$LD_LIBRARY_PATH
./build/test
```

等价于：

```bash
./build/test ./climb0325_2.rv1126.rknn ./images ./results
```

命令行参数格式：

```bash
./build/test [model_path] [image_dir] [output_dir]
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_path` | `climb0325_2.rv1126.rknn` | RKNN 模型路径 |
| `image_dir` | `./images` | 输入图片目录 |
| `output_dir` | `./results` | 结果图片输出目录 |

示例：

```bash
./build/test ./climb0325_2.rv1126.rknn ./images ./results
./build/test /data/models/yolo.rknn /data/images /data/results
```

## 输出结果

程序会在终端打印：

- 模型路径、输入目录、输出目录
- 模型输出 tensor 名称、格式、类型和维度
- 模型输入 tensor 原始维度
- 每张图片的候选框数量
- 最终检测框的类别、置信度和坐标

结果图片保存格式：

```text
results/<原文件名>_result.bmp
```

例如：

```text
images/test1.bmp
results/test1_result.bmp
```

## 后处理参数

当前后处理逻辑位于 `test.cpp`：

- 输入预处理：letterbox 缩放，灰色填充，RGB888。
- 输出假设：3 个 YOLO 风格输出 feature map。
- Anchors：

```cpp
{{10,13},{16,30},{33,23}}
{{30,61},{62,45},{59,119}}
{{116,90},{156,198},{373,326}}
```

- stride：`8`、`16`、`32`
- 目标置信度阈值：`0.85`
- NMS 阈值：`0.3`

如果模型的 anchors、类别数、输出顺序或输出 layout 与这里不一致，需要同步修改 `test.cpp` 中的后处理逻辑。

## PC mock / 兼容说明

`compat/` 目录里有一组 mock 头文件，用于在没有 Rockchip 硬件和 RKNN Runtime 的 PC 环境中模拟 RKNN/RGA 接口，方便验证图片读取、resize、后处理和画框流程。

当前 `CMakeLists.txt` 使用的是 `libs/aarch64` 下的真实 RKNN/RGA 头文件和库，并不是 mock 构建配置。如果要在 PC 上纯模拟运行，需要调整 CMake 的 include/link 配置，让源码包含 `compat/`，并去掉真实 RKNN/RGA 动态库链接。

## 常见问题

### 1. 提示无法打开模型

检查模型文件是否存在，或显式传入模型路径：

```bash
./build/test /path/to/model.rknn ./images ./results
```

### 2. 提示无法打开输入目录

检查第二个参数是否是存在的目录。默认目录是：

```text
./images
```

### 3. RGA 预处理失败

优先检查：

- 当前设备是否支持 RGA。
- `librga.so` 是否能被运行时找到。
- `LD_LIBRARY_PATH` 是否包含 `libs/aarch64/lib`。
- 输入图片是否可以被 `stb_image` 正常读取。

### 4. 没有检测框输出

可能原因：

- 模型与当前 anchors 不匹配。
- 模型类别数或输出 layout 与后处理代码不匹配。
- 置信度阈值 `0.85` 过高。
- 输入图片内容不属于模型训练类别。

可以先降低 `test.cpp` 中 `post_process` 调用处的目标置信度阈值，再观察候选框数量变化。

### 5. 编译时报 `stbi_image_stbi_image_free` 未定义

源码末尾释放图片内存处如果出现这个报错，检查是否存在如下拼写：

```cpp
stbi_image_stbi_image_free(bmp_data);
```

应改为：

```cpp
stbi_image_free(bmp_data);
```

## 开发提示

- 如果更换模型，优先确认输入 tensor 维度、输出 tensor 数量、输出 layout、anchors 和类别数。
- 如果输出类别需要显示名称，可以在 `Box.class_id` 后增加类别名表。
- 如果需要保存 JSON/TXT 检测结果，可以在画框循环中同步写出 `class_id/conf/x1/y1/x2/y2`。
