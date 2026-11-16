"""Small locked CUDA and NF4 validation; run as a module from project root."""
import json
import platform
import torch
from v2.gpulock import gpu_lock


def main():
    with gpu_lock('ws2-env-check'):
        import transformers, accelerate, bitsandbytes as bnb
        import numpy, sklearn, safetensors, huggingface_hub, pytest
        report = dict(python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      device=torch.cuda.get_device_name(0), arch_list=torch.cuda.get_arch_list())
        assert report['cuda_available'] and 'sm_120' in report['arch_list']
        torch.manual_seed(437)
        a, b = torch.randn(16, 16), torch.randn(16, 16)
        err = ((a.cuda() @ b.cuda()).cpu() - a @ b).abs().max().item()
        assert err < 1e-4, err
        linear = torch.nn.Linear(64, 32, bias=False)
        quant = bnb.nn.Linear4bit(64, 32, bias=False, quant_type='nf4', compute_dtype=torch.float16)
        quant.load_state_dict(linear.state_dict())
        quant = quant.to('cuda')
        assert quant.weight.quant_state is not None
        out = quant(torch.randn(2, 64, device='cuda', dtype=torch.float16))
        assert torch.isfinite(out).all()
        report.update(matmul_max_abs_error=err, nf4_quantized=True,
                      nf4_output_shape=list(out.shape), bitsandbytes=bnb.__version__,
                      transformers=transformers.__version__, status='PASS')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
