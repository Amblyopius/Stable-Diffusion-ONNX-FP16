# Copyright 2022 Dirk Moerenhout. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not,
# see <https://www.gnu.org/licenses/>.
#
# *****
# NOTE this was originally derived from:
# https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
#
# Original file released under Apache License, Version 2.0
# *****
#
# Version history
# v1.2 First fully working version converting unet to fp16
# v2.0 Refactored + enabled conversion to fp16 for Text Encoder
# v2.1 Support for safetensors
# v2.2 Reduce visible warnings
# v3.0 You can now provide an alternative VAE
# v3.1 Align with diffusers 0.12.0
# v4.0 Support ckpt conversion (--> renamed to conv_sd_to_onnx.py)
# v5.0 Use ONNX Runtime Transformers for model optimisation

import warnings
import argparse
import os
import shutil
from pathlib import Path
import json
import tempfile

import torch
from torch.onnx import export
import safetensors

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from diffusers.models import AutoencoderKL
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt

# To improve future development and testing, warnings should be limited to what is somewhat useful
# Truncation warnings are expected as part of FP16 conversion and should not be shown
warnings.filterwarnings('ignore','.*will be truncated.*')
# We are ignoring prim::Constant type related warnings
warnings.filterwarnings('ignore','.*The shape inference of prim::Constant type is missing.*')

# ONNX Runtime Transformers offers ONNX model optimisation
# It does not directly support DirectML but we can use a custom class
# Based on onnx_model_unet.py in ONNX Runtime Transformers
from onnx import ModelProto
from onnxruntime.transformers.onnx_model_unet import UnetOnnxModel

class UnetOnnxModelDML(UnetOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize UNet ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)

    def optimize(self, enable_shape_inference=False):
        if not enable_shape_inference:
            self.disable_shape_inference()
        self.fuse_layer_norm()
        self.preprocess()
        self.postprocess()

def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
):
    '''export a PyTorch model as an ONNX model'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export(
        model,
        model_args,
        f=output_path.as_posix(),
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

@torch.no_grad()
def convert_to_fp16(
    model_path
):
    '''Converts an ONNX model on disk to FP16'''
    model_dir=os.path.dirname(model_path)
    # Breaking down in steps due to Windows bug in convert_float_to_float16_model_path
    onnx.shape_inference.infer_shapes_path(model_path)
    fp16_model = onnx.load(model_path)
    fp16_model = convert_float_to_float16(
        fp16_model, keep_io_types=True, disable_shape_infer=True
    )
    # clean up existing tensor files
    shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    # save FP16 model
    onnx.save(fp16_model, model_path)

@torch.no_grad()
def convert_models(pipeline: StableDiffusionPipeline, output_path: str, opset: int, fp16: bool, notune: bool):
    '''Converts the individual models in a path (UNET, VAE ...) to ONNX'''

    output_path = Path(output_path)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    textenc_path=output_path / "text_encoder" / "model.onnx"
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        output_path=textenc_path,
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    if fp16:
        textenc_model_path = str(textenc_path.absolute().as_posix())
        convert_to_fp16(textenc_model_path)
    del pipeline.text_encoder

    # UNET
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size,
                unet_sample_size).to(device=device, dtype=dtype),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=unet_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    del pipeline.unet

    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = os.path.dirname(unet_model_path)
    unet = onnx.load(unet_model_path)
    # clean up existing tensor files
    shutil.rmtree(unet_dir)
    os.mkdir(unet_dir)

    optimizer = UnetOnnxModelDML(unet, 0, 0)
    if not notune:
        optimizer.optimize()
        optimizer.topological_sort()

    # collate external tensor files into one
    onnx.save_model(
        optimizer.model,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    if fp16:
        convert_to_fp16(unet_model_path)
    del unet, optimizer

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample,
        return_dict)[0].sample()
    onnx_export(
        vae_encoder,
        model_args=(
            torch.randn(1, vae_in_channels, vae_sample_size,
                vae_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample", "return_dict"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_out_channels = vae_decoder.config.out_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size,
                unet_sample_size).to(device=device, dtype=dtype),
            False,
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample", "return_dict"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae

    # SAFETY CHECKER
    # NOTE:
    # Safety checker is excluded because it is a resource hog and you'd be turning it off anyway
    # I'm not a legal expert but IMHO you are still bound by the model's license after conversion
    # Check the license of the model you are converting and abide by it

    safety_checker = None
    feature_extractor = None

    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=safety_checker is not None,
    )

    onnx_pipeline.save_pretrained(output_path)
    print("ONNX pipeline saved to", output_path)

    del pipeline
    del onnx_pipeline
    _ = OnnxStableDiffusionPipeline.from_pretrained(output_path,
        provider="DmlExecutionProvider")
    print("ONNX pipeline is loadable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub). "
            "Or the path to a local checkpoint saved in .ckpt or .safetensors."
        )
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output model."
    )

    parser.add_argument(
        "--vae_path",
        default="",
        type=str,
        help=(
            "Path to alternate VAE `diffusers` checkpoint to import and convert (either local or on the Hub). "
            "Works only when converting from diffusers format."
        )
    )

    parser.add_argument(
        "--opset",
        default=15,
        type=int,
        help="The version of the ONNX operator set to use.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export Text Encoder and UNET in mixed `float16` mode"
    )

    parser.add_argument(
        "--notune",
        action="store_true",
        help="Turn off tuning UNET with ONNX Runtime Transformers"
    )

    parser.add_argument(
        "--attention-slicing",
        choices={"auto","max"},
        type=str,
        help=(
            "Attention slicing reduces VRAM needed, off by default. Set to auto or max. "
            "WARNING: max implies --notune"
        )
    )

    parser.add_argument(
        "--clip-skip",
        choices={2,3,4},
        type=int,
        help="Add permanent clip skip to ONNX model."
    )

    parser.add_argument(
        "--diffusers-output",
        type=str,
        help="Directory to dump a pre-conversion copy in diffusers format in."
    )

    parser.add_argument(
        "--ckpt-original-config-file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture."
    )

    parser.add_argument(
        "--ckpt-image-size",
        default=None,
        type=int,
        help="The image size that the model was trained on. Typically 512 or 768"
    )

    parser.add_argument(
        "--ckpt-prediction_type",
        default=None,
        type=str,
        help="Prediction type the model was trained on. 'epsilon' for SD v1.X and SD v2 Base, 'v-prediction' for SD v2"
    )

    parser.add_argument(
        "--ckpt-pipeline_type",
        default=None,
        type=str,
        help="The pipeline type. If `None` pipeline will be automatically inferred."
    )

    parser.add_argument(
        "--ckpt-extract-ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        )
    )

    parser.add_argument(
        "--ckpt-num-in-channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )

    parser.add_argument(
        "--ckpt-upcast-attention",
        action="store_true",
        help="Whether the attention computation should always be upcasted. Necessary when running SD 2.1"
    )

    args = parser.parse_args()

    dtype=torch.float32
    device = "cpu"
    if args.model_path.endswith(".ckpt") or args.model_path.endswith(".safetensors"):
        pl = load_pipeline_from_original_stable_diffusion_ckpt(
            checkpoint_path=args.model_path,
            original_config_file=args.ckpt_original_config_file,
            image_size=args.ckpt_image_size,
            prediction_type=args.ckpt_prediction_type,
            model_type=args.ckpt_pipeline_type,
            extract_ema=args.ckpt_extract_ema,
            scheduler_type="pndm",
            num_in_channels=args.ckpt_num_in_channels,
            upcast_attention=args.ckpt_upcast_attention,
            from_safetensors=args.model_path.endswith(".safetensors")
        )
    else:
        pl = StableDiffusionPipeline.from_pretrained(args.model_path,
            torch_dtype=dtype).to(device)

    if args.vae_path:
        with tempfile.TemporaryDirectory() as tmpdirname:
            pl.save_pretrained(tmpdirname)
            if args.vae_path.endswith('/vae') and not os.path.isdir(args.vae_path):
                vae = AutoencoderKL.from_pretrained(args.vae_path[:-4],subfolder='vae')
            else:
                vae = AutoencoderKL.from_pretrained(args.vae_path)
            pl = StableDiffusionPipeline.from_pretrained(tmpdirname,
                torch_dtype=dtype, vae=vae).to(device)

    if args.clip_skip:
        with tempfile.TemporaryDirectory() as tmpdirname:
            pl.save_pretrained(tmpdirname)
            confname=f"{tmpdirname}/text_encoder/config.json"
            with open(confname, 'r', encoding="utf-8") as f:
                clipconf = json.load(f)
                clipconf['num_hidden_layers'] = clipconf['num_hidden_layers']-args.clip_skip+1
            with open(confname, 'w', encoding="utf-8") as f:
                json.dump(clipconf, f, indent=1)
            pl = StableDiffusionPipeline.from_pretrained(tmpdirname,
                torch_dtype=dtype).to(device)

    blocktune=False
    if args.attention_slicing:
        if args.attention_slicing == "max":
            blocktune=True
            print ("WARNING: attention_slicing max implies --notune")
        pl.enable_attention_slicing(args.attention_slicing)

    if args.diffusers_output:
        pl.save_pretrained(args.diffusers_output)

    convert_models(pl, args.output_path, args.opset, args.fp16, args.notune or blocktune)
    