#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from flulens.model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out", type=str, default="/Users/rik/projects/FluLens/models/flulens.onnx")
    args = parser.parse_args()

    device = get_device()
    model = build_model(num_classes=2).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
