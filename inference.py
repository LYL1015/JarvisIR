import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed
from datetime import datetime
import numpy as np
import megfile
from src.mrrhf.rlhf_engine import RewardEngine
from torch.utils.tensorboard import SummaryWriter
import random
import csv
import tqdm


def build_model(args=None,
                model_architecture=None,
                from_checkpoint=None):
    
    if model_architecture is None:
        model_architecture = args.model_architecture

    if from_checkpoint is None:
        from_checkpoint = args.from_checkpoint
    
    if model_architecture=="llava":
        # half precision, same as demo_gradio.py (fp32 needs ~32GB for the 8B model)
        model = LlavaForConditionalGeneration.from_pretrained(
                from_checkpoint,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(from_checkpoint)

        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        tokenizer.padding_side = 'right'
        
        # freeze parameters
        model.vision_tower.requires_grad_(False)
        model.multi_modal_projector.requires_grad_(True)

        model.language_model.requires_grad_(False)

        return model, image_processor, tokenizer

    else:
        assert "Please add this model architeacture in build_model.py!"

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--from_checkpoint', type=str, default=None, help='Path to the model checkpoint for inference')
parser.add_argument('--image_folder', type=str, default=None, help='Folder containing images for inference')
parser.add_argument('--template', type=str, default='llama_3')
parser.add_argument('--model_architecture', type=str, default='llava')
parser.add_argument('--max_generation_length_of_sampling', type=int, default=384)
parser.add_argument('--max_training_samples_num', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_train_split_ratio', type=float, default=1.0)
parser.add_argument('--save_folder', type=str, default='data/inference/result', help='Folder to save the results')
parser.add_argument('--per_device_train_batch_size', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

parser.add_argument('--save_result_folder', type=str, default='eval_results')
parser.add_argument('--output_dir', type=str, default='eval_results')


args = parser.parse_args()
set_random_seed(args.seed)

os.makedirs(args.save_folder, exist_ok=True)
results = []
device = torch.device("cuda")
writer_config = SummaryWriter(os.path.join(args.output_dir, str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))))
reward_engine = RewardEngine(path=args.save_folder, writer_config=None, device=device)

model, image_processor, tokenizer = build_model(args=args)
processor = AutoProcessor.from_pretrained(args.from_checkpoint)

tokenizer.padding_side = 'right'
model.to(device)
model.eval()

generation_kwargs={
        "top_k": 100,
        "top_p": 0.95,
        "do_sample": True,
        "num_return_sequences": 1,
        "temperature": 1
}


# Available image restoration tasks and their corresponding models
all_tasks = " {denoise: [scunet, restormer], lighten: [retinexformer_fivek, hvicidnet, lightdiff], \
                derain: [idt, turbo_rain, s2former], defog:[ridcp, kanet], \
                desnow:[turbo_snow, snowmaster], super_resolution: [real_esrgan], \
            }"

# Various prompt templates for querying the LLM about image degradation and restoration tasks
prompts_query2 = [
    f"Considering the image's degradation, suggest the required tasks with explanations, and identify suitable tools for each task. Options for tasks and tools include: {all_tasks}.",
    f"Given the image's degradation, outline the essential tasks along with justifications, and choose the appropriate tools for each task from the following options: {all_tasks}.",
    f"Please specify the tasks required due to the image's degradation, explain the reasons, and select relevant tools for each task from the provided options: {all_tasks}.",
    f"Based on the image degradation, determine the necessary tasks and their reasons, along with the appropriate tools for each task. Choose from these options: {all_tasks}.",
    f"Identify the tasks required to address the image's degradation, including the reasons for each, and select tools from the options: {all_tasks}.",
    f"Considering the degradation observed, list the tasks needed and their justifications, then pick the most suitable tools for each task from these options: {all_tasks}.",
    f"Evaluate the image degradation, and based on that, provide the necessary tasks and reasons, along with tools chosen from the options: {all_tasks}.",
    f"With respect to the image degradation, outline the tasks needed and explain why, selecting tools from the following list: {all_tasks}.",
    f"Given the level of degradation in the image, specify tasks to address it, include reasons, and select tools for each task from: {all_tasks}.",
    f"Examine the image's degradation, propose relevant tasks and their explanations, and identify tools from the options provided: {all_tasks}.",
    f"Based on observed degradation, detail the tasks required, explain your choices, and select tools from these options: {all_tasks}.",
    f"Using the image's degradation as a guide, list the necessary tasks, include explanations, and pick tools from the provided choices: {all_tasks}.",
    f"Assess the image degradation, provide the essential tasks and reasons, and select the appropriate tools for each task from the options: {all_tasks}.",
    f"According to the image's degradation, determine which tasks are necessary and why, choosing tools for each task from: {all_tasks}.",
    f"Observe the degradation in the image, specify the needed tasks with justifications, and select appropriate tools from: {all_tasks}.",
    f"Taking the image degradation into account, specify tasks needed, provide reasons, and choose tools from the following: {all_tasks}.",
    f"Consider the image's degradation level, outline the tasks necessary, provide reasoning, and select suitable tools from: {all_tasks}.",
    f"Evaluate the degradation in the image, identify tasks required, explain your choices, and pick tools from: {all_tasks}.",
    f"Analyze the image degradation and suggest tasks with justifications, choosing the best tools from these options: {all_tasks}.",
    f"Review the image degradation, and based on it, specify tasks needed, provide reasons, and select tools for each task from: {all_tasks}."
]


for step, image_name in tqdm.tqdm(enumerate(os.listdir(args.image_folder))):
    # RewardEngine resolves relative image paths against its root (save_folder here),
    # so pass an absolute path; otherwise it looks for save_folder/image_folder/image_name
    image_path = os.path.abspath(os.path.join(args.image_folder, image_name))
    instruction = prompts_query2[random.randint(0, len(prompts_query2)-1)]
    prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    res = tokenizer(
            prompt,
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )
    image = Image.open(image_path).convert("RGB")
    image_outputs = image_processor(image)

    res_all = model.generate(pixel_values=torch.tensor(image_outputs['pixel_values'][0]).unsqueeze(0).to(device, model.dtype), input_ids=torch.LongTensor(res['input_ids']).unsqueeze(0).to(device), \
                                    attention_mask=torch.LongTensor(res["attention_mask"]).unsqueeze(0).to(device), \
                                    max_new_tokens=args.max_generation_length_of_sampling, **generation_kwargs,\
                                    pad_token_id=tokenizer.eos_token_id) # eos_token_id: 128001
    res_text = processor.decode(res_all[0])
    res_text = res_text[res_text.find('<answer>'):]
    res_text = res_text[:res_text.find("<|eot_id|>")]
    _, score = reward_engine.get_reward(image_path, res_text, passcheck=True)
    print(len(score))
    results.append({"image_name": image_name, "qalign_score":score[0], "maniqa_score":score[1],"musiq_score":score[2],"clipiqa_score":score[3], "math.exp(-niqe_score / 10.0)":score[4], "response": res_text, "instruction": instruction})

save_path = os.path.join(args.save_folder, "scores.csv")

with megfile.smart_open(save_path, 'w', newline='') as f:
    fieldnames = ["image_name", "qalign_score", "maniqa_score", "musiq_score", "clipiqa_score", "math.exp(-niqe_score / 10.0)", "response", "instruction"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Saved group CSV for {save_path}, length： {len(results)}")








