from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info

import pano_qwen_erp.register  # noqa: F401
from swift.model import get_model_processor
from transformers import AutoProcessor


@dataclass(frozen=True)
class SmokeCase:
    name: str
    model_id: str
    model_type: str
    processor_id: str | None = None


DEFAULT_CASES = (
    SmokeCase("qwen2.5-vl", "optimum-intel-internal-testing/tiny-random-qwen2.5-vl", "pano_qwen2_5_vl"),
    SmokeCase("qwen3-vl", "tiny-random/qwen3-vl", "pano_qwen3_vl"),
    SmokeCase("qwen3.5", "tiny-random/qwen3.5", "pano_qwen3_5", processor_id="Qwen/Qwen3.5-0.8B"),
)


def _shape_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape_tree(x) for x in obj]
    return type(obj).__name__


def _split_visual_outputs(outputs: Any) -> tuple[Any, Any]:
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output, getattr(outputs, "deepstack_features", None)
    if isinstance(outputs, tuple) and outputs:
        if all(isinstance(x, torch.Tensor) for x in outputs):
            return outputs, None
        if len(outputs) >= 2:
            return outputs[0], outputs[1]
    raise TypeError(f"Unsupported visual output type: {type(outputs)}")


def _build_test_image(path: Path) -> None:
    image = Image.new("RGB", (512, 256), color=(22, 28, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 512, 120), fill=(110, 150, 210))
    draw.rectangle((0, 120, 512, 256), fill=(70, 88, 65))
    draw.rectangle((80, 70, 180, 150), fill=(215, 170, 95))
    draw.rectangle((330, 85, 470, 180), fill=(155, 95, 105))
    draw.line((0, 126, 512, 126), fill=(240, 240, 240), width=3)
    draw.ellipse((18, 20, 58, 60), fill=(255, 215, 90))
    image.save(path)


def _build_inputs(case: SmokeCase, processor, image_path: Path) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Briefly describe this panoramic image."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if "pixel_values" not in inputs and case.processor_id:
        fallback_processor = AutoProcessor.from_pretrained(case.processor_id, trust_remote_code=True)
        inputs = fallback_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        processor = fallback_processor
    return processor, {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _run_case(case: SmokeCase, image_path: Path, max_new_tokens: int) -> None:
    print(f"\n=== {case.name} ===")
    model, processor = get_model_processor(
        case.model_id,
        model_type=case.model_type,
        use_hf=True,
        load_model=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        attn_impl="eager",
    )
    model.eval()
    print(f"model_id={case.model_id}")
    print(f"model_type={case.model_type}")
    print(f"erp_output_adapter={type(getattr(model, 'erp_output_adapter', None)).__name__}")
    print(f"erp_patch_adapter={type(getattr(model, 'erp_patch_adapter', None)).__name__}")
    print(f"erp_merger_adapter={type(getattr(model, 'erp_merger_adapter', None)).__name__}")
    print(f"erp_pos_mode={os.getenv('PANO_ERP_POS_MODE', 'paper')}")
    print(f"erp_stage={os.getenv('PANO_ERP_STAGE', 'output')}")
    print(f"erp_target={os.getenv('PANO_ERP_TARGET', 'both')}")

    processor, inputs = _build_inputs(case, processor, image_path)

    with torch.no_grad():
        image_outputs = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        pooler_output, deepstack_features = _split_visual_outputs(image_outputs)
        print(f"image_pooler_shapes={_shape_tree(pooler_output)}")
        if deepstack_features is not None:
            print(f"deepstack_shapes={_shape_tree(deepstack_features)}")

        forward_outputs = model(**inputs, return_dict=True)
        print(f"logits_shape={tuple(forward_outputs.logits.shape)}")

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        print(f"generation={decoded[0]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test pano Qwen ERP model loading and minimal inference.")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of cases to run: qwen2.5-vl qwen3-vl qwen3.5")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    selected = {name.strip().lower() for name in (args.only or [])}
    cases = [case for case in DEFAULT_CASES if not selected or case.name.lower() in selected]
    if not cases:
        raise ValueError("No smoke-test cases selected.")

    os.environ.setdefault("USE_HF", "1")
    os.environ.setdefault("PANO_ERP_POS_MODE", "paper")
    os.environ.setdefault("PANO_ERP_STAGE", "output")
    os.environ.setdefault("PANO_ERP_TARGET", "both")

    with tempfile.TemporaryDirectory(prefix="pano-qwen-smoke-") as tmp_dir:
        image_path = Path(tmp_dir) / "erp_smoke.png"
        _build_test_image(image_path)
        for case in cases:
            _run_case(case, image_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
