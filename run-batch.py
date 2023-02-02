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

# We need sys for argv
import sys
# We need os.path for isdir, isfile
import os.path
# Our settings are in json format
import json
# To be safe we force gc to lower RAM pressure
import gc
# We want to replace the text encoder in the pipeline
import functools
# Numpy is used to provide a random generator
import numpy
# We need to load images for img2img
# We want to save data to PNG
from PIL import Image, PngImagePlugin

# The pipelines
from diffusers import OnnxStableDiffusionPipeline
from diffusers import OnnxStableDiffusionImg2ImgPipeline
# The schedulers
from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler
)

# Support special text encoders
import lpw_pipe

# Default settings
defSettings = {
    "width": 512,
    "height": 512,
    "steps": 30,
    "stepslist": [],
    "scale": 7.5,
    "scalelist":[],
    "seed":0,
    "seedend":0,
    "seedlist":[],
    "task": "txt2img",
    "model":"sd2_1-fp16",
    "prompt": "",
    "promptlist":[],
    "negative_prompt": "",
    "textenc": "standard",
    "scheduler": "pndm",
    "schedulerlist": [],
    "strength": 0.9,
    "strengthlist": []
}

args = sys.argv
for proj in args[1:] :
    print("Running project "+proj)
    # Check for directory
    if os.path.isdir(proj):
        if os.path.isfile(proj+"/settings.json"):
            with open(proj+"/settings.json", encoding="utf-8") as confFile:
                projSettings=json.load(confFile)
            # Merge dictionaries with project settings taking precedence
            runSettings = defSettings | projSettings
            # We need prompts
            prereqmet=len(runSettings['prompt'])>0 or len(runSettings['promptlist'])>0
            # We need a model
            model="model/"+runSettings['model']
            prereqmet=prereqmet and os.path.isfile(model+"/unet/model.onnx")
            # We need a start image to do img2img
            if runSettings['task']=="img2img":
                infile=proj+"/input.png"
                prereqmet = prereqmet and os.path.isfile(infile)
            if prereqmet:
                sched = {
                    "ddim": DDIMScheduler.from_pretrained(model, subfolder="scheduler"),
                    "deis": DEISMultistepScheduler.from_pretrained(model, subfolder="scheduler"),
                    "dpms_ms": DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler"),
                    "dpms_ss": DPMSolverSinglestepScheduler.from_pretrained(model, subfolder="scheduler"),
                    "euler_anc": EulerAncestralDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                    "euler": EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                    "heun": HeunDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                    "kdpm2": KDPM2DiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                    "lms": LMSDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
                    "pndm": PNDMScheduler.from_pretrained(model, subfolder="scheduler")
                }
                if runSettings['task']=="img2img":
                    init_image = Image.open(infile).convert("RGB")
                    pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                        model,
                        provider="DmlExecutionProvider",
                        revision="onnx",
                        scheduler=sched['pndm'],
                        safety_checker=None,
                        feature_extractor=None
                    )
                else:
                    pipe = OnnxStableDiffusionPipeline.from_pretrained(
                        model,
                        provider="DmlExecutionProvider",
                        revision="onnx",
                        scheduler=sched['pndm'],
                        safety_checker=None,
                        feature_extractor=None
                    )
                if runSettings['textenc'] == "lpw":
                    pipe._encode_prompt = functools.partial(lpw_pipe._encode_prompt, pipe)
                generator = numpy.random
                # Set schedulers for projects
                if len(runSettings['schedulerlist'])==0:
                    schedulerlist=[runSettings['scheduler']]
                else:
                    schedulerlist=runSettings['schedulerlist']
                # Set seeds for project
                if len(runSettings['seedlist'])==0:
                    if runSettings['seed']>runSettings['seedend']:
                        runSettings['seedend']=runSettings['seed']
                    seedlist=range(runSettings['seed'],runSettings['seedend']+1)
                else:
                    seedlist=runSettings['seedlist']
                # Set steps for project
                if len(runSettings['stepslist'])==0:
                    stepslist=[runSettings['steps']]
                else:
                    stepslist=runSettings['stepslist']
                # Set guidance scales for project
                if len(runSettings['scalelist'])==0:
                    scalelist=[runSettings['scale']]
                else:
                    scalelist=runSettings['scalelist']
                # Set prompts for project
                if len(runSettings['promptlist'])==0:
                    promptlist=[runSettings['prompt']]
                else:
                    promptlist=runSettings['promptlist']
                # Set strengths for project
                if len(runSettings['strengthlist'])==0:
                    strengthlist=[runSettings['strength']]
                else:
                    strengthlist=runSettings['strengthlist']
                imgnr=len(schedulerlist)*len(promptlist)*len(seedlist)*len(stepslist)*len(scalelist)*len(strengthlist)
                imgdone=0
                for scheduler in schedulerlist:
                    if not sched[scheduler]:
                        scheduler="pndm"
                    pipe.scheduler=sched[scheduler]
                    promptnum=0
                    for prompt in promptlist:
                        for seed in seedlist:
                            for steps in stepslist:
                                for scale in scalelist:
                                    for strength in strengthlist:
                                        if runSettings['task']=="img2img":
                                            filename=(
                                                f"{proj}/result-p{promptnum}-seed{seed}-steps-{steps}-"+
                                                f"{scheduler}-scale-"+str(scale).replace(".","_")+
                                                "-strength-"+str(strength).replace(".","_")+".png"
                                            )
                                        else:
                                            filename=(
                                                f"{proj}/result-p{promptnum}-seed{seed}-steps-{steps}-"+
                                                f"{scheduler}-scale-"+str(scale).replace(".","_")+".png"
                                            )
                                        if not os.path.isfile(filename):
                                            generator.seed(seed)
                                            if runSettings['task']=="img2img":
                                                image = pipe(
                                                    image=init_image,
                                                    strength=strength,
                                                    prompt=prompt,
                                                    negative_prompt=runSettings['negative_prompt'],
                                                    num_inference_steps=steps,
                                                    guidance_scale=scale,
                                                    generator=generator).images[0]
                                            else:
                                                image = pipe(
                                                    prompt=prompt,
                                                    negative_prompt=runSettings['negative_prompt'],
                                                    height=runSettings['height'],
                                                    width=runSettings['width'],
                                                    num_inference_steps=steps,
                                                    guidance_scale=scale,
                                                    generator = generator).images[0]
                                            metadata = PngImagePlugin.PngInfo()
                                            metadata.add_text("Generator","Stable Diffusion ONNX https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16")
                                            metadata.add_text("SD Model (local name)",model)
                                            metadata.add_text("SD Prompt",prompt)
                                            metadata.add_text("SD Negative Prompt",runSettings['negative_prompt'])
                                            metadata.add_text("SD Scheduler",scheduler)
                                            metadata.add_text("SD Steps",str(steps))
                                            metadata.add_text("SD Guidance Scale",str(scale))
                                            image.save(filename, pnginfo = metadata)
                                        else:
                                            print("Skipping existing image!")
                                        imgdone+=1
                                        print(f"Finished {imgdone}/{imgnr}")
                        promptnum+=1
                del pipe
                gc.collect()
            else:
                print("Minimum requirements not met! Skipping")
        else:
            print("Settings not found! Skipping")
    else:
        print("Path not found! Skipping")
