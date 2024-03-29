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

# We need regular expressions support
import re
# We need argparse for handling command line arguments
import argparse
# We need os.path for isdir
import os.path
# Numpy is used to provide a random generator
import numpy
# Needed to set session options
import onnxruntime as ort


from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Directory in current location to load model from",
    )

    parser.add_argument(
        "--size",
        default=512,
        type=int,
        required=False,
        help="Width/Height of the picture, defaults to 512, use 768 when appropriate",
    )

    parser.add_argument(
        "--steps",
        default=30,
        type=int,
        required=False,
        help="Scheduler steps to use",
    )

    parser.add_argument(
        "--scale",
        default=7.5,
        type=float,
        required=False,
        help="Guidance scale (how strict it sticks to the prompt)"
    )

    parser.add_argument(
        "--prompt",
        default="a dog on a lawn with the eifel tower in the background",
        type=str,
        required=False,
        help="Text prompt for generation",
    )

    parser.add_argument(
        "--negprompt",
        default="blurry, low quality",
        type=str,
        required=False,
        help="Negative text prompt for generation (what to avoid)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Seed for generation, allows you to get the exact same image again",
    )
    
    parser.add_argument(
        "--fixeddims",
        action="store_true",
        help="Pass fixed dimensions to ONNX Runtime. Test purposes only, NOT VRAM FRIENDLY!",
    )

    parser.add_argument(
        "--cpu-textenc", "--cpuclip",
        action="store_true",
        help="Load Text Encoder on CPU to save VRAM"
    )

    parser.add_argument(
        "--cpuvae",
        action="store_true",
        help="Load VAE on CPU, this will always load the Text Encoder on CPU too"
    )

    args = parser.parse_args()

    VAECPU = TECPU = False
    if args.cpuvae:
        VAECPU = TECPU = True
    if args.cpu_textenc:
        TECPU=True

    if match := re.search(r"([^/\\]*)[/\\]?$", args.model):
        fmodel = match.group(1)
    generator=numpy.random
    imgname="testpicture-"+fmodel+"_"+str(args.size)+".png"
    if args.seed is not None:
        generator.seed(args.seed)
        imgname="testpicture-"+fmodel+"_"+str(args.size)+"_seed"+str(args.seed)+".png"

    if  os.path.isdir(args.model+"/unet"):
        height=args.size
        width=args.size
        
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False

        if args.fixeddims:
            sess_options.add_free_dimension_override_by_name("unet_sample_batch", 2)
            sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
            sess_options.add_free_dimension_override_by_name("unet_sample_height", 64)
            sess_options.add_free_dimension_override_by_name("unet_sample_width", 64)
            sess_options.add_free_dimension_override_by_name("unet_timestep_batch", 1)
            sess_options.add_free_dimension_override_by_name("unet_ehs_batch", 2)
            sess_options.add_free_dimension_override_by_name("unet_ehs_sequence", 77)
        
        num_inference_steps=args.steps
        guidance_scale=args.scale
        prompt = args.prompt
        negative_prompt = args.negprompt
        if TECPU:
            cputextenc=OnnxRuntimeModel.from_pretrained(args.model+"/text_encoder")
            if VAECPU:
                cpuvae=OnnxRuntimeModel.from_pretrained(args.model+"/vae_decoder")
                pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                    provider="DmlExecutionProvider", text_encoder=cputextenc, vae_decoder=cpuvae,
                    vae_encoder=None)
            else:
                pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                    provider="DmlExecutionProvider", text_encoder=cputextenc)
        else:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model,
                provider="DmlExecutionProvider", sess_options=sess_options)
        image = pipe(prompt, width, height, num_inference_steps, guidance_scale,
                            negative_prompt,generator=generator).images[0]
        image.save(imgname)
    else:
        print("model not found")
