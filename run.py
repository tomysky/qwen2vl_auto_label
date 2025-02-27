import os
import shutil
import time
import warnings
import torch
warnings.filterwarnings("ignore")
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="cuda", quantization_config=quantization_config
)

processor = AutoProcessor.from_pretrained(model_dir)

files = os.listdir("splite_image")
files.sort(key=lambda x:int(x.split('_')[0]))
for file_path_ in files:
    file_path = "splite_image/"+file_path_
    file_path2 = file_path_.replace(".jpg","")
    print("file:",file_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": file_path,
                },
                {"type": "text", "text": "图片里做主要的物体是什么？不用给我分析过程，只要结果。返回格式：《安全帽》"},
            ],
        }
    ]
    start = time.time()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    if len(output_text)>0:
        new_path = "reg_images/{}_{}.jpg".format(file_path2,output_text[0].replace("《","").replace("》",""))
        print("new path:",new_path)
        try:
            shutil.copy(file_path,new_path)
        except Exception as e:
            print("err:",e)

    print("cost:",time.time()-start)
    print("***********")