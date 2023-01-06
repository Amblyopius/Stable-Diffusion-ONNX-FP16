# Stable Diffusion on AMD using ONNX FP16 and DirectML

This repository contains a conversion tool, some examples, and instructions on how to set up Stable Diffusion with ONNX models for use with AMD GPUs.
This may work on other DirectML devices too, but it's hard to predict if performance will be satisfactory.

This focuses specifically on making it easy to get FP16 models. When using FP16 the VRAM footprint is significantly reduced and speed goes up.

It's all fairly straightforward but It helps to be comfortable with command line

You can use these instructions to convert models to FP16 and then use them in any tool that allows you to load ONNX models.
We'll demonstrate this by downloading and setting up ONNXDiffusersUI specifically for use with our installation (no need to follow the ONNXDiffusersUI setup).

## Set up

First make sure you have Python 3.10 installed. You can get it here: https://www.python.org/downloads/  
**NOTE:** Don't install 3.11 just yet cause not every prerequisite will be available if you do!

If you don't have git, get it here: https://gitforwindows.org/

Create a directory somewhere which can contain all your code.  
Open the commandline (Powershell or Command Prompt) and change into the directory you will use.

Do the following:
```
pip install virtualenv
python -m venv sd_env
sd_env\scripts\activate
python -m pip install --upgrade pip
pip install numpy==1.23.5
pip install transformers diffusers torch ftfy spacy scipy
pip install gradio OmegaConf
pip install --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly-directml
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
python -m pip install --upgrade pip
pip install numpy==1.23.5
pip install transformers diffusers torch ftfy spacy scipy safetensors
pip install onnx onnxconverter_common onnxruntime-directml
```

This will be your environment when you're converting models from diffusers to ONNX.
I prefer to keep these separate because of some conflicting libraries.
Feel free to combine them if you know what you're doing.

Download diffusers_to_onnx_optim.py from this repository and put it in your directory

Now first make sure you have an account on https://huggingface.co/  
When you do make sure to create a token on https://huggingface.co/settings/tokens  
And then on the commandline login using following command
```
huggingface-cli login
```

Now you're ready to download and convert models. Start with the basics and do:
```
mkdir model
python diffusers_to_onnx_optim.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./model/sd2_1base-fp32" 
python diffusers_to_onnx_optim.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./model/sd2_1base-fp16" --fp16
```

You now have 2 models. These are geared towards creating 512x512 images. Get test-txt2img.py from the repository.  
Your environment needs to be sd_env and not sd_env conv to run as otherwise you'll see poor performance or it may just create noise rather than images.  
You can do this by having a second command line window open. Just remember activation is done like this:
```
sd_env\scripts\activate
```

Now we'll run our test script twice (get it from this repository):
```
python test-txt2img.py --model "model\sd2_1base-fp32" --size 512 --seed 0
python test-txt2img.py --model "model\sd2_1base-fp16" --size 512 --seed 0
```

You should now have 2 similar pictures. Note that there'll be differences between FP32 and FP16. But FP16 should not be specifically worse than FP32.
The accuracy just shifts things a bit, but it may just as well shift them for the better. With the default prompt/seed I personally like the FP16 better.

Next let's try do 768x768. This requires your card to have enough VRAM but it does run fine on for example 12GB VRAM. Interested in feedback on how it does on 8GB!  
First make sure you're back on the sd_env_conv environment and then do:
```
python diffusers_to_onnx_optim.py --model_path "stabilityai/stable-diffusion-2-1" --output_path "./model/sd2_1-fp16" --fp16
```

Here we aren't bothering with FP32 because it just requires too much VRAM. Once downloaded we'll just run our test again (in the sd_env environment of course):
```
python test-txt2img.py --model "model\sd2_1-fp16" --size 768 --seed 0
```

Now that we've got everything working and we can create pictures, let's get a GUI. We'll use ONNXDiffusersUI but make it so it doesn't break our workflow.  
The active environment should be sd_env, we already installed the required python modules to support the UI.
Let's first get the repository and copy what we need:
```
git clone https://github.com/azuritecoin/OnnxDiffusersUI
copy OnnxDiffusersUI\onnxUI.py .
copy OnnxDiffusersUI\lpw_pipe.py .
```
That's it, you're now ready to run the UI
```
python onnxUI.py
```
It'll take some time to load and then in your browser you can go to http://127.0.0.1:7860 (only accessible on the host you're running it).  
If you're done you can go back to the CMD window and press Ctrl+C and it will quit.

Note that it expects your models to be in the model directory (which is why we put them there in the instructions).  
You can find your history and all the pictures you created in the directory called output.

## FAQ
### Why are you using ORT Nightly?
The release schedule for ONNX Runtime is quite long and as a result the speed difference between ORT Nightly and the official release is massive.
While there's some risk there's a bug in ORT Nightly, it is just not worth throwing away the performance benefit for the tiny additional guarantee you get from running the official release.

### Do the converted models work with other ONNX DirectML based implementations?
While not tested extensively: yes they should! They are not full FP16, at the interface level they are the same as FP32.
They are completely valid drop in replacements and transparently run in FP16 on ORT DirectML.
This makes it possible to run both FP16 and FP32 models with the exact same code.

### Can I convert non-official models?
Yes, as long as the models have the diffusers format too (not just ckpt). Some suggestions:
- https://huggingface.co/wavymulder/Analog-Diffusion
- https://huggingface.co/Linaqruf/anything-v3.0
- https://huggingface.co/prompthero/openjourney

### Does this work for inpainting / img2img?
Yes, it has been tested on the inpainting models and it works fine. Just like with txt2img, replacement is transparent as the interface is FP32.
Additional example scripts may be added in the future to demonstrate it in code.

### This is still too slow / taxing on my VRAM
Make sure to close as many applications as possible when running FP32 or 768x768 FP16 models.
On my 6700XT I can do 768x768 at 1.2s/it but only if I close all applications.
If I don't close enough applications, it very quickly goes beyond 2s/it.

Also consider following https://github.com/nod-ai/SHARK which provides accelerated ML on AMD via MLIR/IREE.
It (currently) lacks features and flexibility but it has a faster and more VRAM efficient Stable Diffusion implementation than we can currently get on ONNX.  
The current motto also is "Things move fast" which means that in a single day you may get both new features and performance boosts. (On my 6700XT SHARK is close to being twice as fast as ONNX FP16!)  
There's also an onnxdiffusers channel on the Discord where you can ask for help if you want to stick to ONNX for a bit longer. We'll convert you to a dedicated SHARK user there.

### Can you share any results?
On my 6700XT I can get Stable Diffusion 2.1 768x768 down to 1.2s/it and 512x512 to 2.5it/s  
Reported working for Vega56 and doing 512x512 at 1.75it/s  
Reported working for RX 480 8GB and doing 512x512 at 1.75s/it  
Reported working for 5600XT 6GB and doing 512x512 at 1.43s/it (about 4x times faster than using ONNX FP32)

### All these model downloads seem to be eating my main drive's disk space?!
This is an unfortunate side effect of where the huggingface library stores its cache by default.  
On your main drive go to your users home directory (C:\users\...) and you'll find a .cache directory and in it a directory called huggingface.  
Point an environment variable HF_HOME towards where you want to have it store things instead.  
(You can probably move the existing directory to a different drive and point HF_HOME towards it but I have not tested this ...)  
Once resolved you can remove the huggingface directory from .cache

