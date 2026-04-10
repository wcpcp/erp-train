from __future__ import annotations

import os
import sys
from types import MethodType
from typing import Iterable, List, Sequence

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pano_qwen_erp.vision_adapter import ERPSphericalPosAdapter

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.model_arch import ModelArch
from swift.model.patcher import patch_get_input_embeddings
from swift.model.models.qwen import Qwen2_5VLLoader, Qwen3VLLoader, Qwen3_5Loader
from swift.template import TemplateType


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _is_tensor_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in value)


def _parse_csv(value: str) -> List[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _get_visual_module(top_model):
    if hasattr(top_model, "visual"):
        return top_model.visual
    if hasattr(top_model, "model") and hasattr(top_model.model, "visual"):
        return top_model.model.visual
    return None


def _make_adapter(
    *,
    embed_dim: int,
    hidden_dim: int,
    feature_dim: int,
    pos_mode: str,
    gate_init: float,
) -> ERPSphericalPosAdapter:
    return ERPSphericalPosAdapter(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        mode=pos_mode,
        gate_init=gate_init,
    )


def _patch_visual_input_embeddings(top_model) -> None:
    visual = _get_visual_module(top_model)
    if visual is not None:
        patch_get_input_embeddings(visual, "patch_embed")


def _attach_erp_adapter(top_model) -> None:
    if hasattr(top_model, "_pano_erp_attached"):
        return

    hidden_dim = _env_int("PANO_ERP_HIDDEN_DIM", 512)
    gate_init = _env_float("PANO_ERP_GATE_INIT", 0.01)
    pos_mode = _env_str("PANO_ERP_POS_MODE", "paper")
    stages = set(_parse_csv(_env_str("PANO_ERP_STAGE", "output")))
    target = _env_str("PANO_ERP_TARGET", "both").lower()
    feature_dim = 4 if pos_mode == "paper" else 10
    valid_stages = {"patch", "merger", "output"}
    if not stages:
        stages = {"output"}
    unknown = stages - valid_stages
    if unknown:
        raise ValueError(f"Unsupported ERP stage(s): {sorted(unknown)}")
    if target not in {"pooler", "deepstack", "both"}:
        raise ValueError(f"Unsupported ERP target: {target}")
    visual = _get_visual_module(top_model)
    if visual is None:
        raise ValueError("Unable to locate visual module for ERP adapter attachment")

    if "output" in stages:
        top_model.erp_output_adapter = _make_adapter(
            embed_dim=int(top_model.config.text_config.hidden_size),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            gate_init=gate_init,
        )
    if "patch" in stages:
        top_model.erp_patch_adapter = _make_adapter(
            embed_dim=int(visual.config.hidden_size),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            gate_init=gate_init,
        )
    if "merger" in stages:
        top_model.erp_merger_adapter = _make_adapter(
            embed_dim=int(visual.config.hidden_size),
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            pos_mode=pos_mode,
            gate_init=gate_init,
        )

    if stages & {"patch", "merger"} and not hasattr(visual, "_pano_origin_forward"):
        visual._pano_origin_forward = visual.forward

        def visual_forward_with_ctx(this, hidden_states, grid_thw=None, **kwargs):
            this._pano_current_grid_thw = grid_thw
            try:
                return this._pano_origin_forward(hidden_states, grid_thw=grid_thw, **kwargs)
            finally:
                this._pano_current_grid_thw = None

        visual.forward = MethodType(visual_forward_with_ctx, visual)

    if "patch" in stages and not hasattr(visual.patch_embed, "_pano_origin_forward"):
        visual.patch_embed._pano_origin_forward = visual.patch_embed.forward

        def patch_embed_with_erp(this, hidden_states):
            outputs = this._pano_origin_forward(hidden_states)
            grid_thw = getattr(visual, "_pano_current_grid_thw", None)
            if grid_thw is None:
                return outputs
            return top_model.erp_patch_adapter(
                outputs,
                grid_thw,
                int(visual.spatial_merge_size),
                token_layout="premerge",
            )

        visual.patch_embed.forward = MethodType(patch_embed_with_erp, visual.patch_embed)

    if "merger" in stages and not hasattr(visual.merger, "_pano_origin_forward"):
        visual.merger._pano_origin_forward = visual.merger.forward

        def merger_with_erp(this, hidden_states):
            grid_thw = getattr(visual, "_pano_current_grid_thw", None)
            if grid_thw is not None:
                hidden_states = top_model.erp_merger_adapter(
                    hidden_states,
                    grid_thw,
                    int(visual.spatial_merge_size),
                    token_layout="premerge",
                )
            return this._pano_origin_forward(hidden_states)

        visual.merger.forward = MethodType(merger_with_erp, visual.merger)

    if "output" in stages and not hasattr(top_model.model, "_origin_get_image_features"):
        top_model.model._origin_get_image_features = top_model.model.get_image_features

        def get_image_features_with_erp(this, pixel_values, image_grid_thw=None, **kwargs):
            try:
                outputs = this._origin_get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)
            except TypeError as e:
                if "return_dict" not in str(e):
                    raise
                kwargs = {k: v for k, v in kwargs.items() if k != "return_dict"}
                outputs = this._origin_get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)
            if image_grid_thw is None:
                return outputs

            spatial_merge_size = int(this.visual.spatial_merge_size)
            if hasattr(outputs, "pooler_output"):
                if target in {"pooler", "both"}:
                    outputs.pooler_output = top_model.erp_output_adapter(
                        outputs.pooler_output,
                        image_grid_thw,
                        spatial_merge_size,
                        token_layout="merged",
                    )

                if target in {"deepstack", "both"} and getattr(outputs, "deepstack_features", None):
                    outputs.deepstack_features = [
                        top_model.erp_output_adapter(
                            feat,
                            image_grid_thw,
                            spatial_merge_size,
                            token_layout="merged",
                        )
                        for feat in outputs.deepstack_features
                    ]
                return outputs

            if _is_tensor_sequence(outputs):
                adapted = (
                    top_model.erp_output_adapter(
                        outputs,
                        image_grid_thw,
                        spatial_merge_size,
                        token_layout="merged",
                    )
                    if target in {"pooler", "both"}
                    else list(outputs)
                )
                return tuple(adapted) if isinstance(outputs, tuple) else adapted

            if isinstance(outputs, tuple) and outputs:
                pooler_output = outputs[0]
                deepstack_features = outputs[1] if len(outputs) > 1 else None
                if target in {"pooler", "both"}:
                    pooler_output = top_model.erp_output_adapter(
                        pooler_output,
                        image_grid_thw,
                        spatial_merge_size,
                        token_layout="merged",
                    )
                if target in {"deepstack", "both"} and deepstack_features is not None:
                    deepstack_features = [
                        top_model.erp_output_adapter(
                            feat,
                            image_grid_thw,
                            spatial_merge_size,
                            token_layout="merged",
                        )
                        for feat in deepstack_features
                    ]
                head = [tuple(pooler_output) if isinstance(pooler_output, list) else pooler_output]
                if len(outputs) > 1:
                    head.append(deepstack_features)
                return tuple(head + list(outputs[2:]))

            return outputs

        top_model.model.get_image_features = MethodType(get_image_features_with_erp, top_model.model)

    top_model._pano_erp_attached = True


class PanoramaQwen3VLLoader(Qwen3VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers import Qwen3VLForConditionalGeneration

        self.auto_model_cls = self.auto_model_cls or Qwen3VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_visual_input_embeddings(model)
        _attach_erp_adapter(model)
        return model


class PanoramaQwen25VLLoader(Qwen2_5VLLoader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.auto_model_cls = self.auto_model_cls or Qwen2_5_VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_visual_input_embeddings(model)
        _attach_erp_adapter(model)
        return model


class PanoramaQwen35Loader(Qwen3_5Loader):
    def get_model(self, model_dir, config, processor, model_kwargs):
        from transformers import Qwen3_5ForConditionalGeneration

        self.auto_model_cls = self.auto_model_cls or Qwen3_5ForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_visual_input_embeddings(model)
        _attach_erp_adapter(model)
        return model


register_model(
    ModelMeta(
        "pano_qwen2_5_vl",
        [
            ModelGroup(
                [
                    Model("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"),
                    Model("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
                    Model("Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct"),
                    Model("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"),
                    Model("Qwen/Qwen2.5-VL-3B-Instruct-AWQ", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"),
                    Model("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"),
                    Model("Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"),
                    Model("Qwen/Qwen2.5-VL-72B-Instruct-AWQ", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"),
                ],
                TemplateType.qwen2_5_vl,
            ),
        ],
        PanoramaQwen25VLLoader,
        model_arch=ModelArch.qwen2_vl,
        template=TemplateType.qwen2_5_vl,
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        requires=["transformers>=4.49", "qwen_vl_utils>=0.0.6", "decord"],
        tags=["vision", "video", "erp"],
    ))


register_model(
    ModelMeta(
        "pano_qwen3_vl",
        [
            ModelGroup(
                [
                    Model("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"),
                    Model("Qwen/Qwen3-VL-2B-Thinking", "Qwen/Qwen3-VL-2B-Thinking"),
                    Model("Qwen/Qwen3-VL-2B-Instruct-FP8", "Qwen/Qwen3-VL-2B-Instruct-FP8"),
                    Model("Qwen/Qwen3-VL-2B-Thinking-FP8", "Qwen/Qwen3-VL-2B-Thinking-FP8"),
                    Model("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
                    Model("Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
                    Model("Qwen/Qwen3-VL-4B-Thinking", "Qwen/Qwen3-VL-4B-Thinking"),
                    Model("Qwen/Qwen3-VL-8B-Thinking", "Qwen/Qwen3-VL-8B-Thinking"),
                    Model("Qwen/Qwen3-VL-4B-Instruct-FP8", "Qwen/Qwen3-VL-4B-Instruct-FP8"),
                    Model("Qwen/Qwen3-VL-4B-Thinking-FP8", "Qwen/Qwen3-VL-4B-Thinking-FP8"),
                    Model("Qwen/Qwen3-VL-8B-Instruct-FP8", "Qwen/Qwen3-VL-8B-Instruct-FP8"),
                    Model("Qwen/Qwen3-VL-8B-Thinking-FP8", "Qwen/Qwen3-VL-8B-Thinking-FP8"),
                    Model("Qwen/Qwen3-VL-32B-Instruct", "Qwen/Qwen3-VL-32B-Instruct"),
                    Model("Qwen/Qwen3-VL-32B-Thinking", "Qwen/Qwen3-VL-32B-Thinking"),
                    Model("Qwen/Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
                    Model("Qwen/Qwen3-VL-32B-Thinking-FP8", "Qwen/Qwen3-VL-32B-Thinking-FP8"),
                ],
                TemplateType.qwen3_vl,
            ),
        ],
        PanoramaQwen3VLLoader,
        model_arch=ModelArch.qwen3_vl,
        template=TemplateType.qwen3_vl,
        architectures=["Qwen3VLForConditionalGeneration"],
        requires=["transformers>=4.57", "qwen_vl_utils>=0.0.14", "decord"],
        tags=["vision", "video", "erp"],
    ))


register_model(
    ModelMeta(
        "pano_qwen3_5",
        [
            ModelGroup(
                [
                    Model("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
                    Model("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B"),
                    Model("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
                    Model("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
                    Model("Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-27B"),
                    Model("Qwen/Qwen3.5-27B-FP8", "Qwen/Qwen3.5-27B-FP8"),
                    Model("Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B-Base"),
                    Model("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B-Base"),
                    Model("Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B-Base"),
                    Model("Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B-Base"),
                ],
                TemplateType.qwen3_5,
            ),
        ],
        PanoramaQwen35Loader,
        model_arch=ModelArch.qwen2_vl,
        template=TemplateType.qwen3_5,
        architectures=["Qwen3_5ForConditionalGeneration"],
        requires=["transformers>=5.2.0", "qwen_vl_utils>=0.0.14", "decord"],
        tags=["vision", "video", "erp"],
    ))
