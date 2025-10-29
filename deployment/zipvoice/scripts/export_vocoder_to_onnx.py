# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from typing import Dict

import torch
import torch.nn as nn
from conv_stft import STFT
from huggingface_hub import hf_hub_download
from vocos import Vocos





def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--vocoder",
        type=str,
        default="vocos",
        choices=["vocos", "bigvgan"],
        help="Vocoder to export",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./vocos_vocoder.onnx",
        help="Output path for onnx model",
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Export to TensorRT engine",
    )
    parser.add_argument(
        "--trt-output-path",
        type=str,
        default="./vocos_vocoder.plan",
        help="Output path for trt engine",
    )
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"], help="Precision for trt engine")
    parser.add_argument("--min-batch-size", type=int, default=1, help="Min batch size for trt engine")
    parser.add_argument("--opt-batch-size", type=int, default=1, help="Opt batch size for trt engine")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size for trt engine")
    parser.add_argument("--min-input-length", type=int, default=1, help="Min input length for trt engine")
    parser.add_argument("--opt-input-length", type=int, default=1000, help="Opt input length for trt engine")
    parser.add_argument("--max-input-length", type=int, default=3000, help="Max input length for trt engine")
    return parser.parse_args()


def get_trt_kwargs_vocoder(
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 8,
    min_input_length: int = 1,
    opt_input_length: int = 1000,
    max_input_length: int = 3000,
) -> Dict:
    """Get keyword arguments for TensorRT for vocoder."""
    feat_dim = 100
    min_shape = (min_batch_size, feat_dim, min_input_length)
    opt_shape = (opt_batch_size, feat_dim, opt_input_length)
    max_shape = (max_batch_size, feat_dim, max_input_length)
    input_names = ["mel"]
    return {
        "min_shape": [min_shape],
        "opt_shape": [opt_shape],
        "max_shape": [max_shape],
        "input_names": input_names,
    }


def convert_onnx_to_trt(trt_model: str, trt_kwargs: Dict, onnx_model: str, dtype: torch.dtype = torch.float16):
    """
    Convert an ONNX model to a TensorRT engine.

    Args:
        trt_model (str): The path to save the TensorRT engine.
        trt_kwargs (Dict): Keyword arguments for TensorRT.
        onnx_model (str): The path to the ONNX model.
        dtype (torch.dtype, optional): The data type to use. Defaults to torch.float16.
    """
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("failed to parse {}".format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs["input_names"])):
        profile.set_shape(
            trt_kwargs["input_names"][i], trt_kwargs["min_shape"][i], trt_kwargs["opt_shape"][i], trt_kwargs["max_shape"][i]
        )
    if dtype == torch.float16:
        tensor_dtype = trt.DataType.HALF
    elif dtype == torch.float32:
        tensor_dtype = trt.DataType.FLOAT
    else:
        raise ValueError("invalid dtype {}".format(dtype))
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


class ISTFTHead(nn.Module):
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.out = None
        self.stft = STFT(fft_len=n_fft, win_hop=hop_length, win_len=n_fft)

    def forward(self, x: torch.Tensor):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        real = mag * torch.cos(p)
        imag = mag * torch.sin(p)
        audio = self.stft.inverse(input1=real, input2=imag, input_type="realimag")
        return audio


class VocosVocoder(nn.Module):
    def __init__(self, vocos_vocoder):
        super(VocosVocoder, self).__init__()
        self.vocos_vocoder = vocos_vocoder
        istft_head_out = self.vocos_vocoder.head.out
        n_fft = self.vocos_vocoder.head.istft.n_fft
        hop_length = self.vocos_vocoder.head.istft.hop_length
        istft_head_for_export = ISTFTHead(n_fft, hop_length)
        istft_head_for_export.out = istft_head_out
        self.vocos_vocoder.head = istft_head_for_export

    def forward(self, mel):
        waveform = self.vocos_vocoder.decode(mel)
        return waveform


def export_VocosVocoder(vocos_vocoder, output_path, verbose):
    vocos_vocoder = VocosVocoder(vocos_vocoder).cuda()
    vocos_vocoder.eval()

    dummy_batch_size = 8
    dummy_input_length = 500

    dummy_mel = torch.randn(dummy_batch_size, 100, dummy_input_length).cuda()

    with torch.no_grad():
        dummy_waveform = vocos_vocoder(mel=dummy_mel)
        print(dummy_waveform.shape)

    dummy_input = dummy_mel

    torch.onnx.export(
        vocos_vocoder,
        dummy_input,
        output_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "input_length"},
            "waveform": {0: "batch_size", 1: "output_length"},
        },
        dynamo=False,
        verbose=verbose,
    )

    print("Exported to {}".format(output_path))


def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device="cpu", hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        raise NotImplementedError("BigVGAN is not supported yet")
        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


if __name__ == "__main__":
    args = get_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    if args.export_trt:
        import tensorrt as trt

    vocoder = load_vocoder(vocoder_name=args.vocoder, device="cpu", hf_cache_dir=None)
    if args.vocoder == "vocos":
        export_VocosVocoder(vocoder, args.output_path, verbose=False)

    if args.export_trt:
        trt_kwargs = get_trt_kwargs_vocoder(
            min_batch_size=args.min_batch_size,
            opt_batch_size=args.opt_batch_size,
            max_batch_size=args.max_batch_size,
            min_input_length=args.min_input_length,
            opt_input_length=args.opt_input_length,
            max_input_length=args.max_input_length,
        )
        dtype = torch.float16 if args.precision == "fp16" else torch.float32
        convert_onnx_to_trt(args.trt_output_path, trt_kwargs, args.output_path, dtype=dtype)
