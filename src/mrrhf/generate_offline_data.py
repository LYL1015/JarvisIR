import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import sys
from utils.model import build_model 

from utils.utils import to_device
from utils.data import build_dataset, DataCollatorPadToMaxLenForPPOTraining, split_dataset, shuffle_dataset
from tqdm import tqdm
import torch
import logging
import re
from transformers import GenerationConfig
import json
device = torch.device("cuda")
from transformers import AutoTokenizer, AutoProcessor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='dataset/CleanBench-Real_80k/rrhf_data.json')
parser.add_argument('--data_debug_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default="offline_data.json")
parser.add_argument('--dataset_names', type=str, default="llava_ppo")
parser.add_argument('--dataset_samples', type=str, default="all")
parser.add_argument('--dataset_concatenate_samples', type=str, default="1")
parser.add_argument('--max_num_image_per_sample', type=int, default=8)
parser.add_argument('--image_folder', type=str, default='dataset/CleanBench-Real_80k/images/')
parser.add_argument('--template', type=str, default='llama_3')
parser.add_argument('--model_architecture', type=str, default='llava')
parser.add_argument('--from_checkpoint', type=str, default='checkpoints/huggingface_sft_llava')
parser.add_argument('--lang_decoder_update', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_train_split_ratio', type=float, default=0.9)
parser.add_argument('--per_device_train_batch_size', type=int, default=16)
parser.add_argument('--max_seq_len', type=int, default=int(2048 - (336 / 14)**2))
parser.add_argument('--candidate_ans_num', type=int, default=3)

args = parser.parse_args()


# log into files
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('candicate_ans_1019.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

save_path = args.save_path
model, image_processor, tokenizer = build_model(args=args)

model = model.to(device)

dataset = build_dataset(
        args.data_path,
        args.data_debug_path,
        args.dataset_names,
        args.dataset_samples,
        args.dataset_concatenate_samples,
        args.max_num_image_per_sample,
        vis_processor=image_processor,
        vis_root=args.image_folder,
        tokenizer=tokenizer,
        template=args.template
    )

# split the dataset into train and evaluation
np_rng = np.random.RandomState(seed=args.seed)
dataset = shuffle_dataset(dataset, np_rng)

train_dataloader = DataLoader(
    dataset,
    batch_size=args.per_device_train_batch_size,
    sampler=RandomSampler(dataset),
    collate_fn=DataCollatorPadToMaxLenForPPOTraining(args.max_seq_len, tokenizer.pad_token_id, image_processor.crop_size),
)




def sampling_llava_bs(actor_model,
            img, lang,
            attention_mask=None,
            group = 5,
            beam_per_group = 1,
            max_new_tokens=384,
            diversity_penalty=2.0,
            image_paths=None,
            processor=None):
    
    generation_config = GenerationConfig(
        num_beam_groups=group,
        diversity_penalty=diversity_penalty,
        num_beams=group*beam_per_group,
        min_length=1,
        num_return_sequences=group
    )
    dict_res = {}
    batch_size = lang.size()[0]
    actor_model.eval()
    count = []
    # batch_infer
    res_all = actor_model.generate(pixel_values=img, input_ids=lang, \
                                   max_new_tokens=max_new_tokens, generation_config=generation_config, \
                                    attention_mask=attention_mask, pad_token_id=processor.eos_token_id)
    
    for index in range(batch_size):
        res = res_all[index*group:index*group+group]
        res_text = processor.batch_decode(res)
        res_text = [r_text[r_text.find('<reason>'):] + processor.eos_token for r_text in res_text]
        res_text = [r_text[:r_text.find('</answer>')+len('</answer>')] for r_text in res_text]

        res_text_set = set()
        for i in range(group):
            res_text_set.add(res_text[i])
        count.append(len(res_text_set))
        dict_res[image_paths[index]] = list(res_text_set)
        # logger.info(f"{image_paths[index]}: {res_text_set}, ")
    return count, dict_res

def filter_tools(tool_sequence):
    pattern = r"\[type:.+?\]:\(model:(.+?)\)"
    return ",".join(re.findall(pattern, tool_sequence))


result_dict = {}
temp = -1
# models
for step, batch in enumerate(tqdm(train_dataloader)):
    batch = to_device(batch, device)
    images = batch["image"].half()
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    batch_size = input_ids.size()[0]
    image_sizes = None

    total_counts = []
    counts, dict_tmp = sampling_llava_bs(model, 
                    images, input_ids, 
                    processor=tokenizer,
                    image_paths=batch["image_path"],
                    group=args.candidate_ans_num)
    total_counts.extend(counts)
    result_dict.update(dict_tmp)

    if len(result_dict) // 500 > temp:
        temp = len(result_dict) // 500
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

with open(save_path, "w", encoding="utf-8") as json_file:
    json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

