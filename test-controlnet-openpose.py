from PIL import Image
import numpy as np
from diffusers import DEISMultistepScheduler
from pipeline_onnx_stable_diffusion_controlnet import OnnxStableDiffusionControlNetPipeline
import onnxruntime as ort

pose_image = Image.open(r"dance_pose.png")

opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern = False

pipe = OnnxStableDiffusionControlNetPipeline.from_pretrained(
    "model/anyv3-fp16-autoslicing-cn_openpose",
    sess_options=opts, 
    provider="DmlExecutionProvider",
)

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
prompt = "1girl, blonde, long dress, dancing, best quality"
seed=25
generator = np.random.RandomState(seed)

images = pipe(    
    prompt,
    pose_image,
    width=512,
    height=512,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=30,
    generator=generator,
).images[0]
images.save("controlnet-openpose-test.png")
