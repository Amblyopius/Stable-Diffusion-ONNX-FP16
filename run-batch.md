# Running Stable Diffusion ONNX DirectML batches

Ever feel like it's a struggle to compare schedulers, guidance scale ... while using a UI?  
Not really interested in coding in Python to resolve it?

Hopefully run-batch.py will make your life a bit easier.

## Set up
Drop run-batch.py where you've installed OnnxDiffusersUI (it'll use the same lwp_pipe.py).

As parameter run-batch.py accepts 1 or more paths where it will then check for the xisting of settings.json.
It'll read settings.json, create a batch of images and save them in the directory it just got the settings from.

In settings.json you can define the following things:
- The model, set with the key 'model'. It will look for a directory with that name in the model subdirectory (just like OnnxDiffusersUI).
- The scheduler, set with the key 'scheduler'. You can also set a list of schedulers to iterate over with the key 'schedulerlist'.
It currently accepts following values: ddim, deis, dpms_ms, dpms_ss, euler_anc, euler, heun, kdpm2, lms, pndm.
If the value is not recognised, it'll switch to pndm. Not all schedulers have been extensively tested and may behave unexpectendly.
- Guidance scale, set with the key 'scale'. You can also set a list of guidance scales to iterate over with the key 'scalelist'
- Iteration steps, set with the key 'steps'. You can also set a list of steps to iterate over with the key 'stepslist'
- Width and height, with the keys 'width' and 'height'
- The seed, set with the key 'seed'. If you want to iterate over seeds you can define the end with the key 'seedend'.
Alternatively, you can provide a list of seeds using the key 'seedlist'.
- The task to perform, set with the key 'task', default is 'txt2img'. Only txt2img has been test enough, but it also supports img2img (more on that below).
- The prompt, set with the key 'prompt'. If you want to iterate over prompts, you can define them with the key 'promptlist'.
- A negative prompt, set with the key 'negative_prompt'. Note that the same negative prompt will apply to all prompts you provided.
- How to parse the prompt, set with the key 'textenc'. Supports 2 values, 'standard' and 'lwp'. Use 'lwp' when you want to use weights and long prompts.

If you are doing img2img there's more options:
- Strength, set with the key 'strength'. Expected to be between 0 to 1. You can iterate over a list by setting 'strengthlist'.

For img2img the directory will also need to contain an image file called input.png

## Example

Compare the results for a prompt with deis and euler at 20, 30 and 40 steps. Using SD 2.1.

```
{
	"model": "sd2_1-fp16",
	"seed": 0,
	"seedend": 10,
	"stepslist": [20,30,40],
	"scale": 8.5,
	"schedulerlist": ["deis","euler"],
	"prompt": "(photo portrait) of a ((beautiful)) (woman) wearing (summer dress) in a park, eyes, detailed, high resolution, prime lens",
	"negative_prompt": "bad quality, low resolution",
	"textenc": "lwp"
}
```
