#!/usr/bin/env python3
"""
Export trained policy to ONNX or TorchScript for deployment
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsl_rl.runners import OnPolicyRunner


def export_onnx(model_path: str, output_path: str, obs_dim: int, action_dim: int):
    """Export policy to ONNX format."""
    runner = OnPolicyRunner(None, train_cfg={}, log_dir="logs/export")
    runner.load(model_path)

    policy = runner.get_inference_policy()
    policy.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Export
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Exported ONNX model to {output_path}")


def export_torchscript(model_path: str, output_path: str):
    """Export policy to TorchScript."""
    runner = OnPolicyRunner(None, train_cfg={}, log_dir="logs/export")
    runner.load(model_path)

    policy = runner.get_inference_policy()
    policy.eval()

    # Trace
    dummy_input = torch.randn(1, runner.obs_dim)
    traced_script_module = torch.jit.trace(policy, dummy_input)
    traced_script_module.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained policy")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--obs-dim", type=int, default=28, help="Observation dimension (28 dims)")
    parser.add_argument("--action-dim", type=int, default=6, help="Action dimension (6 joints including gripper)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.model)[0]
        args.output = f"{base}.{args.format}"

    if args.format == "onnx":
        export_onnx(args.model, args.output, args.obs_dim, args.action_dim)
    else:
        export_torchscript(args.model, args.output)


if __name__ == "__main__":
    main()
