
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import sys

prompt = sys.argv[1]

print("Prompt: ", prompt)

# Set up openVLA model and processor
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

#prompt = "In: What action should the robot take to open the top drawer?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt).to("cuda:0", dtype=torch.bfloat16)
actions = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(actions)