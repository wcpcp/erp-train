# Pano Qwen ERP SFT

面向 `ERP 全景图` 的 `Qwen2.5-VL / Qwen3-VL / Qwen3.5` 微调实验脚手架。

这个项目的目标不是重写 Qwen 主干，而是以**低侵入**方式，把球面几何先验接进现有多模态模型，方便做下面两类实验：

- 结构实验：把 ERP 球面位置编码接到不同层位
- 训练实验：做球面旋转一致性训练与相关消融

当前项目基于官方 `ms-swift`，本地固定在 `third_party/ms-swift`，版本为 `v4.0.3`。

## 1. 当前核心思路

这个仓库当前实现的是：

- 保留 Qwen 原有 attention、RoPE、多模态主干
- 额外引入一个 `ERP spherical position adapter`
- 用很小的 gate 和零初始化尾层，让新增模块一开始接近 identity
- 通过环境变量切换位置编码挂点，方便做 ablation

当前默认策略是：

- `PANO_ERP_POS_MODE=paper`
- `PANO_ERP_STAGE=output`
- `PANO_ERP_TARGET=both`

也就是：

- 球面位置特征使用最接近 `PanoVGGT` 的 4 维编码
- 在 `get_image_features` 输出后加位置偏置
- 对 `pooler_output` 和 `deepstack_features` 都做适配

如果你追求更稳的 baseline，推荐先用：

```bash
PANO_ERP_STAGE=output
PANO_ERP_TARGET=pooler
```

## 2. 当前已经实现的功能

### 2.1 支持的模型

已经接入：

- `pano_qwen2_5_vl`
- `pano_qwen3_vl`
- `pano_qwen3_5`

对应基座模型可直接用：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`

推荐优先顺序：

1. `Qwen3-VL-4B-Instruct`
2. `Qwen2.5-VL-3B-Instruct`
3. `Qwen3.5-4B`

### 2.2 已实现的位置编码功能

当前实现了三种**结构挂点**：

- `patch`
- `merger`
- `output`

以及一种输出分支选择：

- `pooler`
- `deepstack`
- `both`

### 2.3 已实现的球面位置特征

当前支持两种位置特征模式：

#### `PANO_ERP_POS_MODE=paper`

最接近 `PanoVGGT` 的实现：

```python
[sin(yaw), cos(yaw), sin(pitch), cos(pitch)]
```

#### `PANO_ERP_POS_MODE=extended`

扩展实验版：

```python
[
  sin(yaw), cos(yaw), sin(pitch), cos(pitch),
  sin(2*yaw), cos(2*yaw), sin(2*pitch), cos(2*pitch),
  cos(pitch), normalized_pitch
]
```

建议：

- 做主实验时先用 `paper`
- `extended` 只在后续扩展消融时再开

### 2.4 已实现的训练脚本

已经提供：

- `scripts/train_qwen2_5_vl_3b_lora.sh`
- `scripts/train_qwen2_5_vl_3b_full.sh`
- `scripts/train_qwen3_vl_4b_lora.sh`
- `scripts/train_qwen3_vl_4b_full.sh`
- `scripts/train_qwen3_vl_8b_lora.sh`
- `scripts/train_qwen3_5_4b_lora.sh`
- `scripts/train_qwen3_5_4b_full.sh`

其中 LoRA 脚本已经会根据 `PANO_ERP_STAGE` 自动设置 `modules_to_save`，保存当前启用的 adapter：

- `erp_patch_adapter`
- `erp_merger_adapter`
- `erp_output_adapter`

### 2.5 已实现的本地 smoke test

脚本：

- `scripts/smoke_test_models.py`

它会检查：

- 自定义 plugin 注册
- 模型加载
- ERP adapter 挂载
- `get_image_features`
- 最小 `forward`
- 最小 `generate`

## 3. 位置编码加在哪里

这是当前项目最重要的结构实验维度。

### 3.1 `PANO_ERP_STAGE=patch`

含义：

- 在 `patch_embed` 输出后
- 进入视觉 blocks 之前
- 给最早期视觉 token 加 ERP 球面位置偏置

逻辑上相当于：

```python
x = patch_embed(pixel_values)
x = x + gate * spherical_pos_embed
x = x + original_pos_embed
x = vision_blocks(x)
```

优点：

- 几何信息进入最早
- 视觉主干从第一层开始就“知道”自己在处理 ERP
- 对 seam、极区、ERP 畸变更敏感

风险：

- 对原视觉主干影响最大
- 比较容易影响原模型通用视觉能力
- 更适合在 baseline 之后做

这里使用的位置网格是：

- `premerge`

也就是：

- 每个最细视觉 token 都按原始 patch 网格计算球面坐标

### 3.2 `PANO_ERP_STAGE=merger`

含义：

- 视觉 blocks 已经跑完
- 但还没进入最终 `merger`
- 在视觉特征压到语言视觉 token 之前加 ERP 位置偏置

逻辑上相当于：

```python
h = vision_blocks_output
h = h + gate * spherical_pos_embed
z = merger(h)
```

优点：

- 不改视觉 blocks 内部计算
- 比 `output` 更早，还能影响 token 如何被 merge
- 比 `patch` 更稳

风险：

- 仍然会影响最终视觉 token 的形成方式
- 风险介于 `patch` 和 `output` 之间

这里使用的位置网格也是：

- `premerge`

也就是：

- merger 看到的输入 token 仍然能一一对应到 ERP 上的局部区域

### 3.3 `PANO_ERP_STAGE=output`

含义：

- 视觉塔和 merger 都已经结束
- 在 `get_image_features` 输出之后
- 对最终视觉 token 再加 ERP 位置偏置

逻辑上相当于：

```python
z = get_image_features(...)
z = z + gate * spherical_pos_embed
```

优点：

- 风险最低
- 最不容易破坏原始 Qwen 能力
- 最适合作为第一版 baseline

风险：

- 只能影响最终视觉 token 语义
- 不能影响视觉 token 在视觉塔中的交互过程

这里使用的位置网格是：

- `merged`

也就是：

- 对最终送进 LLM 的 merged visual token 计算 ERP 中心角

### 3.4 `PANO_ERP_TARGET`

这个变量只在 `output` 阶段有意义。

可选项：

- `pooler`
- `deepstack`
- `both`

解释：

- `pooler`：只改最终视觉 token
- `deepstack`：只改 `Qwen3-VL` 的 deepstack 分支
- `both`：两者都改

如果你希望实验更稳，建议：

```bash
PANO_ERP_STAGE=output
PANO_ERP_TARGET=pooler
```

## 4. 当前代码结构

### 核心文件

- `src/pano_qwen_erp/erp_geometry.py`
  - 负责把 ERP 网格转换成球面特征
  - 同时支持 `premerge` 和 `merged` 两种 token 布局

- `src/pano_qwen_erp/vision_adapter.py`
  - 负责把球面特征投影成和 visual token 同维度的向量
  - 然后用 gated residual 的方式加回 token

- `src/pano_qwen_erp/register.py`
  - 负责注册 `ms-swift` 模型
  - 负责把 adapter 挂到 `patch / merger / output` 不同阶段

- `src/pano_qwen_erp/data/prepare_sft.py`
  - 把原始 ERP 标注转成 `ms-swift` 可训练 JSONL

- `docs/panovggt_to_qwen.md`
  - 论文思路到当前工程实现的迁移分析

## 5. 环境搭建

建议先准备好和服务器一致的 CUDA PyTorch，然后：

```bash
cd /Users/wcp/code/erp_data_pipeline/pano_qwen_erp_sft
bash scripts/bootstrap.sh
source .venv/bin/activate
```

如果你要用 `Qwen3.5`，当前还需要升级 `transformers`：

```bash
source .venv/bin/activate
python -m pip install --upgrade 'git+https://github.com/huggingface/transformers.git'
```

必要时补：

```bash
pip install decord
```

## 6. 本地验证

### 全量 smoke test

```bash
cd /Users/wcp/code/erp_data_pipeline/pano_qwen_erp_sft
source .venv/bin/activate
python scripts/smoke_test_models.py --max-new-tokens 2
```

### 指定阶段验证

#### output

```bash
PANO_ERP_STAGE=output python scripts/smoke_test_models.py --only qwen3-vl --max-new-tokens 1
```

#### patch

```bash
PANO_ERP_STAGE=patch python scripts/smoke_test_models.py --only qwen3-vl --max-new-tokens 1
```

#### merger

```bash
PANO_ERP_STAGE=merger python scripts/smoke_test_models.py --only qwen3-vl --max-new-tokens 1
```

已本地验证通过的组合包括：

- `Qwen3-VL + output`
- `Qwen3-VL + patch`
- `Qwen3-VL + merger`
- `Qwen2.5-VL + merger`
- `Qwen2.5-VL / Qwen3-VL / Qwen3.5 + output`

## 7. 数据格式

这个项目支持两种数据接入方式：

1. 直接把本地 `json/jsonl` 作为 `--dataset` 输入
2. 可选地先用 `prepare_sft.py` 做轻量标准化

如果你已经有 `messages + images` 这种多模态数据，**推荐直接训练，不需要额外在外面写转换代码**。

### 7.0 你当前这种数据可以直接训练

像下面这种数据：

- 顶层有 `messages`
- 顶层有 `images`
- `messages[*].content` 是 OpenAI 风格的 `list[{"type":"text"}, {"type":"image"}]`

可以直接传给当前训练脚本，不需要你先写一层外部转换程序。

也就是说，这种格式：

```json
{
  "id": "scene_00001:caption:0006",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a multimodal assistant specialized in ERP panoramic image understanding."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Give a more complete visual description of the oval white table with clear legs."
        },
        {
          "type": "image"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "An oval-shaped white dining table with clear acrylic legs."
        }
      ]
    }
  ],
  "images": [
    "/workspace/data_dir/.../panoImage_1600.jpg"
  ]
}
```

直接就可以用于：

```bash
TRAIN_DATA=/abs/path/train.json \
VAL_DATA=/abs/path/val.json \
PANO_ERP_STAGE=output \
PANO_ERP_TARGET=pooler \
bash scripts/train_qwen3_vl_4b_lora.sh
```

或者：

```bash
TRAIN_DATA=/abs/path/train.jsonl \
VAL_DATA=/abs/path/val.jsonl \
PANO_ERP_STAGE=output \
PANO_ERP_TARGET=pooler \
bash scripts/train_qwen3_vl_4b_lora.sh
```

### 7.1 直接训练的标准格式

训练器最终吃的是 `ms-swift` 的标准多模态格式。文件可以是：

- `.json`
- `.jsonl`

最小格式如下：

```jsonl
{"messages":[{"role":"user","content":"<image>Describe the geometry."},{"role":"assistant","content":"..."}],"images":["/abs/path/pano.jpg"]}
```

### 7.2 直接支持 OpenAI 风格多模态 `messages`

你现在这种格式也可以直接训练：

```json
[
  {
    "id": "scene_00001:caption:0006",
    "messages": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": "You are a multimodal assistant specialized in ERP panoramic image understanding."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Give a more complete visual description of the oval white table with clear legs."
          },
          {
            "type": "image"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "An oval-shaped white dining table with clear acrylic legs."
          }
        ]
      }
    ],
    "images": [
      "/workspace/data_dir/data_user/public_data/360video/Realsee3D/real_world_data/scene_00001/viewpoints/1753781394/panoImage_1600.jpg"
    ]
  }
]
```

这类格式当前框架会直接兼容：

- `messages[*].content` 可以是字符串
- 也可以是 `list[{"type":"text"}, {"type":"image"}]`
- 顶层 `images` 按顺序提供图像路径
- 顶层 `id` 会保留
- 额外字段如 `erp_meta` 也可以保留

也就是说，你完全可以直接这样训练：

```bash
TRAIN_DATA=/abs/path/train.json \
VAL_DATA=/abs/path/val.json \
PANO_ERP_STAGE=output \
PANO_ERP_TARGET=pooler \
bash scripts/train_qwen3_vl_4b_lora.sh
```

或者：

```bash
TRAIN_DATA=/abs/path/train.jsonl \
VAL_DATA=/abs/path/val.jsonl \
PANO_ERP_STAGE=output \
PANO_ERP_TARGET=pooler \
bash scripts/train_qwen3_vl_4b_lora.sh
```

### 7.3 关于 `<image>` 和 `{"type":"image"}`

两种写法都可以：

- 字符串形式：`"<image>Describe the panorama."`
- 结构化形式：`[{"type":"text","text":"Describe the panorama."},{"type":"image"}]`

如果你已经在用 OpenAI 风格的多模态 `messages`，建议继续保持这种结构化写法，不需要人为改成纯字符串。

### 7.4 关于 `erp_meta`

你也可以在样本里保留：

```json
"erp_meta": {
  "projection": "ERP",
  "anchor": {"yaw_deg": 0.0, "pitch_deg": 0.0},
  "camera_frame": "+x right, +y up, +z forward"
}
```

注意：

- 当前结构里的球面位置编码不依赖这个字段
- 它主要用于数据记录、后续分析、以及可选的 prompt 注入
- 如果你想把这些元信息拼到用户问题里，可以再用下面的可选转换脚本

### 7.5 可选转换脚本

如果你的原始数据不是标准 `messages` 格式，或者你希望自动把 `erp_meta` 注入到文本 prompt 里，可以用：

```bash
source .venv/bin/activate
python -m pano_qwen_erp.data.prepare_sft \
  --input /abs/path/raw_erp_annotations.jsonl \
  --output /abs/path/train.jsonl \
  --system-prompt "You are a panoramic 3D reasoning assistant."
```

如果要把 `erp_meta` 自动拼到 user prompt：

```bash
python -m pano_qwen_erp.data.prepare_sft \
  --input /abs/path/raw_erp_annotations.jsonl \
  --output /abs/path/train.jsonl \
  --inject-erp-metadata
```

这个脚本现在同样支持你这种 `messages + images + content list` 的 OpenAI 风格数据，它会：

- 保留 `messages`
- 保留 `images`
- 保留 `id`
- 保留 `erp_meta`
- 如果开启 `--inject-erp-metadata`，会把元信息加到第一个 user text 段中，而不会破坏 `{"type":"image"}` 占位

样例数据：

- `examples/data/pano_erp_sft_sample.jsonl`

## 8. 训练

### 推荐第一组实验

```bash
cd /Users/wcp/code/erp_data_pipeline/pano_qwen_erp_sft
source .venv/bin/activate

TRAIN_DATA=/abs/path/train.jsonl \
VAL_DATA=/abs/path/val.jsonl \
OUTPUT_DIR=/abs/path/output/qwen3_vl_4b_erp \
PANO_ERP_POS_MODE=paper \
PANO_ERP_STAGE=output \
PANO_ERP_TARGET=pooler \
bash scripts/train_qwen3_vl_4b_lora.sh
```

### LoRA 与 Full

当前支持两类：

- `SFT + LoRA`
- `full-parameter SFT`

说明：

- `SFT` 是训练目标
- `LoRA` 是参数高效微调方式
- `full` 是全参数微调

Full 脚本当前默认：

- `FREEZE_VIT=false`
- `FREEZE_ALIGNER=false`

如果你要更保守的 full run，可以覆盖：

```bash
FREEZE_VIT=true FREEZE_ALIGNER=true bash scripts/train_qwen3_vl_4b_full.sh
```

## 9. 建议的结构消融

建议至少跑下面这几组：

### Baseline

```bash
PANO_ERP_STAGE=output
PANO_ERP_TARGET=pooler
PANO_ERP_POS_MODE=paper
```

### 更强的 output 版

```bash
PANO_ERP_STAGE=output
PANO_ERP_TARGET=both
PANO_ERP_POS_MODE=paper
```

### merger 消融

```bash
PANO_ERP_STAGE=merger
PANO_ERP_POS_MODE=paper
```

### patch 消融

```bash
PANO_ERP_STAGE=patch
PANO_ERP_POS_MODE=paper
```

### 组合实验

```bash
PANO_ERP_STAGE=merger,output
PANO_ERP_TARGET=pooler
PANO_ERP_POS_MODE=paper
```

### 位置特征扩展实验

```bash
PANO_ERP_STAGE=output
PANO_ERP_TARGET=pooler
PANO_ERP_POS_MODE=extended
```

## 10. 训练方式上的主创新点

当前项目在结构上是“低侵入球面位置适配”，在训练上最推荐的创新方向是：

### 球面旋转一致性训练

核心目标：

- 让模型学会 `ERP 只是球面的一种参数化表示`
- 而不是把 ERP 图像上的固定横向位置当成内容本身

最推荐的三类任务：

1. 语义不变性
   - caption 一致性
   - 场景语义一致性

2. 几何等变性
   - 左右前后
   - yaw / pitch 方位
   - 相对方向

3. 参考系归一化一致性
   - 给定 anchor 或 front 定义后
   - 图像旋转但参考系同步变化时
   - 答案应保持一致

最建议先从：

- `yaw-only`

开始做，因为：

- 最干净
- 最少插值伪影
- 最适合 ERP

## 11. rope 相关建议

当前仓库**还没有**改 visual RoPE。

这是刻意的，因为直接替换原 RoPE 风险太高。

如果后面要做 rope 方向实验，建议顺序是：

### 第一优先：residual attention bias

形式：

```python
score_ij = q_i k_j / sqrt(d) + alpha * b_sph(i, j)
```

优点：

- 不替换原 RoPE
- 不直接改 q/k 主体
- 风险最低

### 第二优先：partial delta-RoPE

形式：

```python
q' = q_rope + alpha * dq_sph(pos)
k' = k_rope + alpha * dk_sph(pos)
```

建议：

- 只在一部分视觉 head dim 上做
- 保持 residual/gated

### 当前不建议

- 直接整体替换 visual RoPE
- 大幅改写 attention 主体

## 12. 当前已实现 vs 未实现

### 已实现

- ERP 球面位置编码
- `paper / extended` 两种特征模式
- `patch / merger / output` 三种结构挂点
- `pooler / deepstack / both` 输出分支控制
- `Qwen2.5-VL / Qwen3-VL / Qwen3.5` 接入
- `LoRA / full SFT` 脚本
- smoke test

### 还未实现

- 球面旋转一致性训练的数据构造器
- residual attention bias
- delta-RoPE
- 几何辅助 loss
- depth / normals / pose side heads

## 13. 项目结构

```text
pano_qwen_erp_sft/
├── docs/
├── examples/
├── scripts/
├── src/pano_qwen_erp/
│   ├── data/
│   ├── erp_geometry.py
│   ├── register.py
│   └── vision_adapter.py
└── third_party/ms-swift/
```

## 14. 最推荐的下一步

如果你现在要继续往前推进，最建议的顺序是：

1. 先把 `output / merger / patch` 三种结构 ablation 跑清楚
2. 在最优结构上加 `yaw-only 球面旋转一致性训练`
3. 最后再做 residual attention bias

如果只是做第一版最稳的实验，建议直接从这个配置起步：

```bash
PANO_ERP_POS_MODE=paper
PANO_ERP_STAGE=output
PANO_ERP_TARGET=pooler
```
