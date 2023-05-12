# Stable Diffusion using ONNX, FP16 and DirectML

This repository contains a conversion tool, some examples, and instructions on how to set up Stable Diffusion with ONNX models.
This was mainly intended for use with AMD GPUs but should work just as well with other DirectML devices (e.g. Intel Arc).
I'd be very interested to hear of any results with Intel Arc.  

**MOST IMPORTANT RECENT UPDATES:**  
**- Realigned with latest version of diffusers, we were forced to switch to torch 2.1 nightly! (Install instructions updated accordingly)**  
**- Added an ONNX ControlNet pipeline (documented in additional section after standard install)**  
**- Added an ONNX Instruct pix2pix pipeline (documented in additional section after standard install)**  
**- Added support for Clip Skip**  
**- ONNX Runtime 1.14 has been released! Thanks to this we now have a significantly simplified installation process.**  
**- I have enabled GitHub discussions: If you have a generic question rather than an issue, start a discussion!**

This focuses specifically on making it easy to get FP16 models. When using FP16, the VRAM footprint is significantly reduced and speed goes up.

It's all fairly straightforward, but It helps to be comfortable with command line.

You can use these instructions to convert models to FP16 and then use them in any tool that allows you to load ONNX models.
We'll demonstrate this by downloading and setting up ONNXDiffusersUI specifically for use with our installation (no need to follow the ONNXDiffusersUI setup).

## Set up

First make sure you have Python 3.10 installed. You can get it here: https://www.python.org/downloads/  
**NOTE:** Don't install 3.11 just yet cause not every prerequisite may be available if you do!

If you don't have git, get it here: https://gitforwindows.org/

Pick a directory that can contain your Stable Diffusion installation (make sure you've the diskspace to store the models).
Open the commandline (Powershell or Command Prompt) and change into the directory you will use.

Start by cloning this repository:
```
git clone https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16
cd Stable-Diffusion-ONNX-FP16
```

Do the following:
```
pip install virtualenv
python -m venv sd_env
sd_env\scripts\activate
python -m pip install --upgrade pip
pip install torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --pre
pip install -r requirements.txt
```

Now first make sure you have an account on https://huggingface.co/  
When you do make sure to create a token on https://huggingface.co/settings/tokens  
And then on the commandline login using following command
```
huggingface-cli login
```

Now you're ready to download and convert models. Before we explain this, just a pointer on future use.  
Whenever you want to make use of this post set up, open a command line, change into the directory and enable the environment.
Say that you installed this on your D: drive in the root. You would open command line and then:
```
d:
cd Stable-Diffusion-ONNX-FP16
sd_env\scripts\activate
```

Remember this for whenver you want to use your installation. Let's now get to the fun part and convert some models:
```
mkdir model
python conv_sd_to_onnx.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./model/sd2_1base-fp32" 
python conv_sd_to_onnx.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./model/sd2_1base-fp16" --fp16
```

You now have 2 models. These are geared towards creating 512x512 images.

Now we'll run our test script twice:
```
python test-txt2img.py --model "model\sd2_1base-fp32" --size 512 --seed 0
python test-txt2img.py --model "model\sd2_1base-fp16" --size 512 --seed 0
```

You should now have 2 similar pictures. Note that there'll be differences between FP32 and FP16. But FP16 should not be specifically worse than FP32.
The accuracy just shifts things a bit, but it may just as well shift them for the better.

Next let's do 768x768. This requires your card to have enough VRAM but we'll make a VRAM friendly version too.
Here we aren't bothering with FP32 because it just requires too much VRAM. 
```
python conv_sd_to_onnx.py --model_path "stabilityai/stable-diffusion-2-1" --output_path "./model/sd2_1-fp16" --fp16
python test-txt2img.py --model "model\sd2_1-fp16" --size 768 --seed 0
```

You should now have a 768x768 picture.

This will work fine on 12GB VRAM and above but 8GB may already be a stretch. The more VRAM friendly version is next.

This method uses less VRAM and will be slightly slower when you're not VRAM limited. But, it'll allow you to use far larger resolutions than standard models.
The output will be slightly different but should not be specifically worse. If you got the VRAM, see how well size 1024 works!

```
python conv_sd_to_onnx.py --model_path "stabilityai/stable-diffusion-2-1" --output_path "./model/sd2_1-fp16-autoslicing" --fp16 --attention-slicing auto
python test-txt2img.py --model "model\sd2_1-fp16-autoslicing" --size 768 --seed 0
```

Now that we've got everything working and we can create pictures, let's get a GUI. We'll use ONNXDiffusersUI but make it so it doesn't break our workflow.  
First we clone the repository:
```
git clone https://github.com/azuritecoin/OnnxDiffusersUI
```
Now we run the UI
```
python OnnxDiffusersUI\onnxUI.py
```
It'll take some time to load and then in your browser you can go to http://127.0.0.1:7860 (only accessible on the host you're running it).  
If you're done you can go back to the CMD window and press Ctrl+C and it will quit.

Note that it expects your models to be in the model directory (which is why we put them there in the instructions).  
You can find your history and all the pictures you created in the directory called output.

If you want to learn more about the UI be sure to visit https://github.com/azuritecoin/OnnxDiffusersUI

## Advanced features
### Support for ControlNet
ControlNet was recently introduced. It allows conditional control on Text-to-Image Diffusion Models. If you want more in-depth information,
get it here: https://github.com/lllyasviel/ControlNet

As it has now been added to Diffusers I've added a fairly "elegant" ONNX implementation.

The idea behind the implementation is:  
- We use the same single tool to convert models  
- We can load the Pipeline from disk by referencing a single model

This has only 1 downside, it is not the most disk friendly solution as you'll get some duplication.
We may eventually have to opt for a different disk layout for ONNX models.

The current implementation consists of a simple demo. More to follow soon!

First let's get ourselves a working model:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_ft_mse-autoslicing-cn_canny" --controlnet_path "lllyasviel/sd-controlnet-canny" --fp16 --attention-slicing auto --vae_path "stabilityai/sd-vae-ft-mse"
```
This model is an SD 1.5 model combined with Controlnet Canny. Now let's run the test script:
```
python test-controlnet-canny.py
```
Once the test is done you'll have an image called controlnet-canny-test.png.
The new image is entirely different but the shape is very similar to the original input image.

You can look at the test-controlnet-canny.py to see how it works.

Next we'll use openpose. Note that the example is a demanding pose that you would ordinarily probably not go for.
Without tweaking this also suffers a bit from bad hands/feet. For the sake of the test I decided to tolerate it.
Let's make a ControlNet OpenPose model:
```
python conv_sd_to_onnx.py --model_path "Linaqruf/anything-v3.0" --output_path "./model/anyv3-fp16-autoslicing-cn_openpose" --controlnet_path "lllyasviel/sd-controlnet-openpose" --fp16 --attention-slicing auto
```
And now let's run the test:
```
python test-controlnet-openpose.py
```
This gives you controlnet-openpose-test.png

As some may wonder where I got the openpose startpoint image from. I used https://zhuyu1997.github.io/open-pose-editor/  
Create the pose.
Press the button underneath height, then download the generated map on the left.
You can further edit it locally to fit the canvas in the way you want it to.

### Support for Instruct pix2pix
Recently a special Stable Diffusion model was released, allowing you to have AI edit images based on instructions.
Make sure you read the original documentation here: https://www.timothybrooks.com/instruct-pix2pix

A pipeline was added to diffusers, but currently Huggingface does not add ONNX equivalents.
In this repository I included the required ONNX pipeline and a basic UI (to simplify testing before it gets added to ONNXDiffusersUI)

You can convert the model using this command (it'll fetch it from huggingface):
```
python conv_sd_to_onnx.py --model_path "timbrooks/instruct-pix2pix" --output_path "./model/ip2p-base-fp16-vae_ft_mse-autoslicing" --vae_path "stabilityai/sd-vae-ft-mse" --fp16 --attention-slicing auto
```
Once converted you can run the included UI like this:
```
python pix2pixUI.py
```
You'll need an image to start from (you can always create one with Stable Diffusion) and then you can test the pipeline.
This first version is _very_ basic and you'll need to save the results (when you want them) using "save image as" in your browser.

### Use alternative VAE
Some models will suggest using an alternative VAE.
It's possible to copy the model.onnx from an existing directory and put it in another one, but you may want to keep the conversion command line you use for reference.
To simplify the task of using an alternative VAE you can now pass it as part of the conversion command.

Say you want to have SD1.5 but with the updated MSE VAE that was released later and is the result of further training. You can do it like this:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_ft_mse" --vae_path "stabilityai/sd-vae-ft-mse" --fp16
```

You can also load a vae from a full model on huggingface. You add /vae to make that clear. Say you need the VAE from Anything v3.0:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_anythingv3" --vae_path "Linaqruf/anything-v3.0/vae" --fp16
```

Or if the model is on your local disk, you can just use the local directory. Say you have stable-diffusion 2.1 base on disk, you could it like this:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_2_1" --vae_path "stable-diffusion-2-1-base/vae" --fp16
```

### Clip Skip
For some models people will suggest using "Clip Skip" for better results. As we can't arbitrarily change this with ONNX, we need to decide on it at model creation.  
Therefore there's --clip-skip which you can set to 2, 3 or 4.  

Example:
```
python conv_sd_to_onnx.py --model_path "Linaqruf/anything-v3.0" --output_path "./model/anythingv3_fp16_cs2" --fp16 --clip-skip 2
```

Clip Skip results in a change to the Text Encoder. 
To stay compatible with other implementations we use the same numbering where 1 is the default behaviour and 2 skips 1 layer.
This ensures that you see similar behaviour to other implementations when setting the same number for Clip Skip.

### Reducing VRAM usage
While FP16 already uses a lot less VRAM, you may still run into VRAM issues. The easiest solution is to load the Text Encoder on CPU rather than GPU. The Text Encoder is only used as part of prompt parsing and not during the iterations.
You can expect some additional latency when the Text Encoder is on CPU, but this will be fairly minor as it is not compute intensive. You also gain more than that back during the iterations if you're near your VRAM limit.
You'll bump into VRAM limits when it is limited (8GB or less), or you're trying to use a 768x768 model.

In test-txt2img.py you can see how this works. You can pass --cpu-textenc and it will load the Text Encoder on CPU. This is how it's done:
```
            cputextenc=OnnxRuntimeModel.from_pretrained(args.model+"/text_encoder")
            pipe = OnnxStableDiffusionPipeline.from_pretrained(args.model, provider="DmlExecutionProvider", text_encoder=cputextenc)
```
You can use this in your own code when needed. OnnxDiffusersUI supports --cpu-textenc too.

In extreme circumstances you can also try to load VAE on CPU. This is likely to be only of use for cards that have limited VRAM. The need to load VAE on CPU can be identified when generation crashes after the steps.
So if it goes through all the steps but then crashes when it needs to save the final image, VAE is your issue. If it crashes before steps is finished, changes to where VAE is loaded are unlikely to make much of a difference.  
**You can pass --cpuvae to test-txt2img.py to load VAE on CPU (this will always also load CLIP on CPU).**  
Note that having VAE loaded on CPU is CPU intensive (far more than CLIP is) and you'll see RAM use spike.

### Conversion of .ckpt / .safetensors
Did your model come as a single file ending in .safetensors or .ckpt? Don't worry, with the 0.12.0 release of diffusers I can now use diffusers to load these directly. I have updated (and renamed) the conversion tool and it 
will convert directly from .ckpt to ONNX.

This is probably the most requested feature as many of you have used https://www.civitai.com/ and have found the conversion process a bit cumbersome.

To properly convert a file you do need a .yaml config file. Ideally this should be included but if not you're advised to try with the v1-inference.yaml included in this repository.  
To convert a model you'd then do:
```
python conv_sd_to_onnx.py --model_path ".\downloaded.ckpt" --output_path "./model/downloaded-fp16" --ckpt-original-config-file downloaded.yaml --fp16
```
If it did not come with a .yaml config file, try with v1-inference.yaml.

If you have a choice between .safetensors and .ckpt, go for .safetensors. In theory a .ckpt file can contain malicious code. I have not seen any reports of this happening but it's better to be safe than sorry.

The conversion tool also has additional parameters you can set when converting from .ckpt/.safetensors. The best way to find all the parameters is by doing:
```
python conv_sd_to_onnx.py --help
```
You should generally not need these but some advanced users may want to have them just in case.

## FAQ
### Do the converted models work with other ONNX DirectML based implementations?
While not tested extensively: yes they should! They are not full FP16, at the interface level they are the same as FP32.
They are completely valid drop in replacements and transparently run in FP16 on ORT DirectML.
This makes it possible to run both FP16 and FP32 models with the exact same code.

### Can I convert non-official models?
You should be able to convert any model. Most of the models can be found on https://huggingface.co/, but you may prefer using https://civitai.com/ instead.
It's generally better to start from a model in diffusers form but if you only have the .ckpt/.safetensors file you now have instructions on how to convert these directly into ONNX.

### Does this work for inpainting / img2img?
Yes, it has been tested on the inpainting models and it works fine. Just like with txt2img, replacement is transparent as the interface is FP32.
Additional example scripts may be added in the future to demonstrate inpainting in code. For now mainly useful for use with OnnxDiffusersUI

### Why is Euler Ancestral not giving me the same image if I provide a seed?
Due to how Euler Ancestral works, it adds noise as part of the scheduler that is apparently non-deterministic when interacting with ONNX diffusers pipeline.
A clean ONNX implementation without diffusers, torch ... would likely be faster and bug free but it's a lot of work and it would not match SHARK.  
Best advice is to live with it and to switch to SHARK as soon as your wished for feature is available there. For more on SHARK see the next answer.

### This is still too slow / taxing on my VRAM
Make sure to close as many applications as possible when running FP32 or 768x768 FP16 models.
On my 6700XT I can do 768x768 at 1.2s/it but only if I close all applications.
If I don't close enough applications, it very quickly goes beyond 2s/it.

Also consider following https://github.com/nod-ai/SHARK which provides accelerated ML on AMD via MLIR/IREE.
It (currently) lacks features and flexibility but it has a faster and more VRAM efficient Stable Diffusion implementation than we can currently get on ONNX.  
The current motto also is "Things move fast" which means that in a single day you may get both new features and performance boosts. (On my 6700XT SHARK is close to being twice as fast as ONNX FP16!)  
There's also an onnxdiffusers channel on the Discord where you can ask for help if you want to stick to ONNX for a bit longer. We'll convert you to a dedicated SHARK user there.

If you are an advanced AMD user, switch to Linux+ROCm. It'll be faster and you can use any torch based solution directly.

### Can you share any results?
On my 6700XT I can get Stable Diffusion 2.1 768x768 down to 1.15s/it and 2.1 base 512x512 to 2.7it/s  
Reported working for Vega56 and doing 512x512 at 1.75it/s  
Reported working for RX 480 8GB and doing 512x512 at 1.75s/it  
Reported working for 5600XT 6GB and doing 512x512 at 1.43s/it (about 4x times faster than using ONNX FP32)

### All these model downloads seem to be eating my main drive's disk space?!
This is an unfortunate side effect of where the huggingface library stores its cache by default.  
On your main drive go to your users home directory (C:\users\...) and you'll find a .cache directory and in it a directory called huggingface.  
Point an environment variable HF_HOME towards where you want to have it store things instead.  
(You can probably move the existing directory to a different drive and point HF_HOME towards it but I have not tested this ...)  
Once resolved you can remove the huggingface directory from .cache
