# Stable Diffusion using ONNX, FP16 and DirectML

This repository contains a conversion tool, some examples, and instructions on how to set up Stable Diffusion with ONNX models.

**These instructions are specifically for people who have only 4GB VRAM**

It's all fairly straightforward, but It helps to be comfortable with command line.

We'll focus on making all of it work within limited VRAM. This will still include a UI.

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

Remember this for whenver you want to use your installation. Let's now get to the fun part and convert a model. This will take some time!
The extra time spend on creating the model is saved back by having it run fine on 4GB VRAM.
```
mkdir model
python conv_sd_to_onnx.py --model_path "stabilityai/stable-diffusion-2-1-base" --output_path "./model/sd2_1base-fp16-maxslicing" --fp16 --attention-slicing max
```

That's your first model. Let's do a test:

```
python test-txt2img.py --model "model\sd2_1base-fp16-maxslicing" --size 512 --seed 0 --cpu-textenc --cpuvae
```

You should now have your first picture in the current directory.

Now that we've got everything working and we can create pictures, let's get a GUI. We'll use ONNXDiffusersUI but make it so it doesn't break our workflow.  
First we clone the repository:
```
git clone https://github.com/azuritecoin/OnnxDiffusersUI
```
Now we run the UI
```
python OnnxDiffusersUI\onnxUI.py --cpu-textenc --cpu-vaedec
```
It'll take some time to load and then in your browser you can go to http://127.0.0.1:7860 (only accessible on the host you're running it).  
If you're done you can go back to the CMD window and press Ctrl+C and it will quit.

Note that it expects your models to be in the model directory (which is why we put them there in the instructions).  
You can find your history and all the pictures you created in the directory called output.

If you want to learn more about the UI be sure to visit https://github.com/azuritecoin/OnnxDiffusersUI

## Advanced features
### Use alternative VAE
Some models will suggest using an alternative VAE.
It's possible to copy the model.onnx from an existing directory and put it in another one, but you may want to keep the conversion command line you use for reference.
To simplify the task of using an alternative VAE you can now pass it as part of the conversion command.

Say you want to have SD1.5 but with the updated MSE VAE that was released later and is the result of further training. You can do it like this:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_ft_mse" --vae_path "stabilityai/sd-vae-ft-mse" --fp16 --attention-slicing max
```

You can also load a vae from a full model on huggingface. You add /vae to make that clear. Say you need the VAE from Anything v3.0:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_anythingv3" --vae_path "Linaqruf/anything-v3.0/vae" --fp16 --attention-slicing max
```

Or if the model is on your local disk, you can just use the local directory. Say you have stable-diffusion 2.1 base on disk, you could it like this:
```
python conv_sd_to_onnx.py --model_path "runwayml/stable-diffusion-v1-5" --output_path "./model/sd1_5-fp16-vae_2_1" --vae_path "stable-diffusion-2-1-base/vae" --fp16 --attention-slicing max
```

### Clip Skip
For some models people will suggest using "Clip Skip" for better results. As we can't arbitrarily change this with ONNX we need to decide on it at model creation.  
Therefore there's --clip-skip which you can set to 2, 3 or 4.  

Example:
```
python conv_sd_to_onnx.py --model_path "Linaqruf/anything-v3.0" --output_path "./model/anythingv3_fp16_cs2" --fp16 --clip-skip 2
```

Clip Skip results in a change to the Text Encoder. To stay compatible with other implementations we use the same numbering where 1 is the default behaviour and 2 skips 1 layer.
This ensures that you see similar behaviour to other implementations when setting the same number for Clip Skip.

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
