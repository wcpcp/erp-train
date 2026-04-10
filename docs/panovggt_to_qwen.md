# From PanoVGGT to Qwen

## 1. What PanoVGGT is really doing

The key ideas in `PanoVGGT` are not "a brand-new transformer is required". The more transferable parts are:

1. `Spherical-aware position embedding`
2. `SO(3)`-consistent panorama augmentation
3. `Permutation-equivariant multi-view reasoning`
4. `Stochastic anchoring`
5. `Over-complete geometry supervision`

The most important observation for adapting this into Qwen is that only the first two can be brought into a standard VLM pipeline almost immediately without damaging pretrained capability.

## 2. Why not change Qwen attention first

For your goal, directly changing Qwen attention is high risk:

- pretrained Qwen weights are tightly coupled to the current architecture
- even small attention-path changes can destabilize inherited capability
- SFT-scale training is usually not enough to recover general-purpose vision-language ability after such surgery

So the safest path is:

- keep the Qwen backbone
- inject ERP geometry through additive or gated side modules
- use data and supervision design to teach spherical 3D reasoning

## 3. Best low-intrusion upgrades

### 3.1 ERP spherical position adapter

This is the main implemented idea in this repo.

Instead of replacing Qwen positional encoding, we:

- read the post-vision visual tokens
- compute ERP-aware angular features from their `(yaw, pitch)` location
- project those features with a tiny MLP
- add them back with a small learned gate

Why this is good:

- it preserves pretrained Qwen weights
- it is easy to train with LoRA
- it makes seam continuity and latitude distortion explicit
- it is reversible and easy to ablate

Compared with the paper, we extend the raw `sin/cos` encoding slightly with:

- second-order harmonics
- latitude area weighting

This gives the model a bit more ERP distortion awareness without changing the backbone.

### 3.2 ERP-consistent data augmentation

This should be done in the data pipeline, not inside the backbone:

- random yaw rotation is mandatory
- pitch / roll augmentation should be controlled, not always maximal
- if annotations include direction, pose, or anchor-based answers, labels must be rotated consistently

Practical recommendation:

- stage 1: yaw-only augmentation for stable training
- stage 2: add moderate pitch / roll
- stage 3: full `SO(3)` augmentation only for tasks whose labels rotate correctly

### 3.3 Anchor-conditioned language supervision

PanoVGGT uses stochastic anchoring to remove global-frame bias. In Qwen SFT, the analogous trick is:

- randomly choose a reference direction or camera frame
- express the answer relative to that anchor

Example:

- "Take the forward ray at yaw=0, pitch=0 as anchor"
- "Use the camera frame: +x right, +y up, +z forward"
- "Answer relative depth and direction in this anchor frame"

This is much better than letting the model produce vague words like "left" or "far" without a defined coordinate system.

### 3.4 Multi-view sidecar aggregator

This is the next upgrade after the current repo.

Do not modify Qwen decoder attention first. Instead:

- encode each panorama independently with the existing vision tower
- insert a small sidecar geometric aggregator between the visual encoder and the language model
- make that sidecar handle cross-view token fusion

This copies the spirit of PanoVGGT alternating attention while keeping Qwen mostly intact.

A practical design is:

- intra-view adapter for ERP distortion correction
- cross-view latent tokens for set aggregation
- fused visual tokens sent into the original Qwen decoder

That is a much lower-risk form of "geometry aggregator" than rewriting the base model.

### 3.5 Auxiliary geometry heads

This is the best stage-two training improvement.

Keep the main task as SFT, but add optional side heads to the visual branch or projector:

- depth bins or metric depth
- surface normal bins
- horizon / vanishing line
- relative yaw / pitch between anchor and queried object
- coarse point-map tokens or voxel occupancy summaries

Why this helps:

- language loss alone is weak for fine 3D geometry
- geometry heads shape the visual features directly
- the LLM part stays intact

The right training order is:

1. warm up with `SFT + ERP adapter`
2. add geometry auxiliary loss on the same adapter branch
3. only then test stronger cross-view modules

## 4. Recommended training roadmap

### Stage A: safest baseline

- Model: `Qwen3-VL-4B-Instruct`
- Training: `LoRA`
- Frozen: original ViT and aligner
- Extra trainable module: `erp_adapter`
- Data: single-image ERP geometry QA / caption / direction / relative depth

Goal:

- teach the model ERP-specific spatial bias
- keep original multimodal competence intact

### Stage B: stronger but still safe

- Model: `Qwen3-VL-8B-Instruct`
- Add `modules_to_save erp_adapter`
- LoRA on language layers
- optionally unfreeze or LoRA the aligner only

Goal:

- improve transfer from visual tokens to language reasoning

### Stage C: geometry-assisted SFT

- keep stage B setup
- add auxiliary heads for depth / normals / relative direction
- train multi-task with a small geometry loss weight

Goal:

- move from "language about geometry" to "features that encode geometry"

### Stage D: multi-view ERP reasoning

- feed multiple panoramas in one sample
- add sidecar set aggregator before the decoder
- use anchor-conditioned outputs and relative-pose tasks

Goal:

- approximate PanoVGGT-style global reasoning without sacrificing Qwen initialization

## 5. What to collect in the dataset

For SFT to work well, do not only collect generic captions.

High-value ERP 3D supervision includes:

- anchor-based object direction
- relative depth ordering
- walkable free-space description
- room / street topology summary
- cross-seam object continuity
- horizon and ground-plane cues
- object pair relation in spherical coordinates
- multi-view relative pose description

If you later have stronger supervision, add:

- monocular depth
- normals
- coarse point clouds
- semantic occupancy

## 6. Final recommendation

If the goal is "improve ERP omnidirectional 3D understanding without hurting original Qwen capability", the best first bet is:

1. start from `Qwen3-VL-4B/8B-Instruct`
2. add the ERP spherical position adapter
3. train with ERP-aware instruction data
4. use anchor-conditioned supervision
5. then optionally add auxiliary geometry heads

This path is much more realistic than modifying Qwen attention early.

## 7. How Qwen2.5-VL fits in

`Qwen2.5-VL` is a strong baseline for the same idea because its multimodal path is architecturally compatible with the adapter insertion point used in this repo.

Useful mental model:

- `Qwen2.5-VL` already has a native vision branch
- it computes visual tokens and injects them into the language side
- it already uses multimodal position handling with 3D position ids
- therefore we do not need to replace its internal position system to add ERP awareness

So the same low-risk strategy still applies:

- keep its original multimodal RoPE and visual pipeline
- add an ERP spherical adapter after the visual encoder output
- let LoRA update the downstream reasoning path

Compared with `Qwen3-VL`:

- `Qwen2.5-VL` is a good baseline for ablation
- `Qwen3-VL` is the more forward-looking main path
- both can share the same ERP adapter design in this repo
