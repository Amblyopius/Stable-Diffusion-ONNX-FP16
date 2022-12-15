# Stable Diffusion on AMD using ONNX FP16

This repository contains a conversion tool and instructions to set up Stable Diffusion with ONNX for use with AMD GPUs.  Focus is on getting the best result on Windows with ONNX Runtime DirectML

It's fairly straightforward but:
- Best to be comfortable with command line
- Best to be able to code basic stuff in Python as initially there'll be limited code provided here

## Set up

First make sure you have Python 3.10 installed. You can get it here: https://www.python.org/downloads/
  **NOTE:** Don't install 3.11 just yet cause not every prerequisite will be available if you do!

If you don't have git get it here: https://gitforwindows.org/

Create a directory somewhere which can contain all your code.
  Open the commandline (Powershell or Command Prompt) and change into the directory you will use.

Do the following:
```
pip install virtualenv
python -m venv sd_env
sd_env\scripts\activate
python -m pip install --upgrade pip
pip install transformers diffusers torch ftfy spacy scipy
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly-directml
```

This will be your environment for when you're creating images.
  Now do:
```
sd_env\scripts\deactivate
```

And then:
```
python -m venv sd_env_conv
sd_env_conv\scripts\activate
pip install transformers diffusers torch ftfy spacy scipy
pip install onnx onnxruntime-directml
```

This will be your environment when you're converting models from diffusers to ONNX.
I prefer to keep these separate because of some conflicting libraries.
Feel free to combine them if you know what you're doing.

Download diffusers_to_onnx_optim.py from this repository and put it in your directory

Now first make sure you have an account on https://huggingface.co/
  When you do make sure to create a token one https://huggingface.co/settings/tokens
  And then on the commandline login using following command
```
huggingface-cli
```

Now you're ready to download and convert models. Start with the basics and do:
```
python diffusers_to_onnx_optim.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./sd2_1base-fp32" 
python diffusers_to_onnx_optim.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./sd2_1base-fp16" --fp16
```

You now have 2 models. These are geared towards creating 512x512 images. Get test-txt2img-512.py from the repository.
  Your environment needs to be sd_env and not sd_env conv to run as otherwise you'll see poor performance.
  You can do this by having a second command line window open. Just remember activation is done like this:
```
sd_env\scripts\activate
```

Now we'll run our test script twice:
```
python test-txt2img.py --model "sd2_1base-fp32" --size 512
python test-txt2img.py --model "sd2_1base-fp16" --size 512
```

You should now have 2 near identical pictures. Note that there'll be differences between FP32 and FP16. But FP16 should not be specifically worse than FP32.

Next let's try do 768x768. This requires your card to have enough VRAM but it does run fine on for example 12GB VRAM. Interested in feedback on how it does on 8GB!
  First make sure you're back on the sd_env_conv environment and then do:
```
python diffusers_to_onnx_optim-v2_0.py --model_path "stabilityai/stable-diffusion-2-1" --output_path "./sd2_1-fp16" --opset 15 --fp16
```

Here we aren't bothering with FP32 because it just requires too much VRAM. Once downloaded we'll just run our test again:
```
python test-txt2img.py --model "sd2_1-fp16" --size 768
```

## FAQ
### Why are you using ORT Nightly?
The release schedule for ONNX Runtime is quite long and as a result the speed difference between ORT Nightly and the official release is massive.
While there's some risk there's a bug in ORT Nightly it is just not worth throwing away the performance benefit for the tiny additional guarantee you get from running the official release.

### Do these models work with other ONNX DirectML based implementations?
While not tested extensively: yes they should! The advantage is also that they are not full FP16, at the interface level they are the same as FP32 and hence completely valid drop in replacements.
