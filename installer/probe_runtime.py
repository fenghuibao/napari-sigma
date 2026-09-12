"""Verify the installed compute runtime; report GPU tests separately from CPU."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    bundle = json.loads((args.resources / "bundle.json").read_text(encoding="utf-8"))
    expected = next(record["version"] for record in bundle["wheels"] if record["name"] == "torch")
    assert torch.__version__ == expected, (torch.__version__, expected)
    if bundle["platform"] == "win-64":
        assert sys.platform == "win32"
        assert torch.backends.cuda.is_built(), "The Windows app must not be CPU-only"
        assert torch.version.cuda == "13.0", torch.version.cuda
    data = torch.arange(27, dtype=torch.float32).reshape(1, 1, 3, 3, 3) / 27
    kernel = torch.ones(1, 1, 3, 3, 3) / 27
    cpu = torch.nn.functional.conv3d(data, kernel, padding=1)
    assert torch.isfinite(cpu).all()
    result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cpu_test": "passed", "cuda_available": torch.cuda.is_available(),
              "cuda_hardware_test": "not run: no usable NVIDIA GPU on this machine"}
    if result["cuda_available"]:
        actual = torch.nn.functional.conv3d(data.cuda(), kernel.cuda(), padding=1)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), cpu, atol=1e-5, rtol=1e-5)
        result.update(cuda_hardware_test="passed", gpu=torch.cuda.get_device_name(),
                      capability=torch.cuda.get_device_capability())
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
