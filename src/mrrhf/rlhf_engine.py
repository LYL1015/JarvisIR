# DeepSpeed Team
import math
import torch
import os
import deepspeed
import sys
from transformers import AdamW
from transformers import get_scheduler
# `utils` lives next to this file; make it importable both when this file is run
# from src/mrrhf (train.sh) and when it is imported as src.mrrhf.rlhf_engine (inference.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.model import build_model
from utils.ds_utils import get_train_ds_config
from utils.utils import get_optimizer_grouped_parameters, print_rank_0
import re
from agent_tools import RestorationToolkit
class DeepSpeedRLHFEngine():
    def __init__(self, actor_model_name_or_path, 
                 actor_tokenizer=None,
                 reward_tokenizer=None,
                 number_dataset=None,
                 args=None,
                 writer_config=None):
        self.args = args
        self.queue_size = 1000
        self.expanding_multiples = 10
        self.reward_queue = torch.tensor([], dtype=torch.float16).cuda()

        self.number_dataset = number_dataset

        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer

        self.actor, self.actor_image_processor, self.actor_tokenizer_new = self._init_actor(
            actor_model_name_or_path)
        self.reward_engine = RewardEngine(path=args.image_folder, writer_config=writer_config, device=self.actor.device)
        
        self.actor_tokenizer_new.padding_side = 'right'
        self.actor_tokenizer_new.padding_side = 'right'
        
    def push_queue(self, reward_scores):
        self.reward_queue = torch.cat((self.reward_queue, reward_scores))
        if len(self.reward_queue) > self.queue_size:
            self.reward_queue = self.reward_queue[-self.queue_size:]
    
    def reward_score_standard(self, reward_scores):
        self.push_queue(reward_scores)
        reward_mean = torch.mean(self.reward_queue)
        reward_std = torch.std(self.reward_queue)
        reward_scores_standard = (reward_scores - reward_mean) / reward_std
        return reward_scores_standard * self.expanding_multiples

    def _init_actor(self, actor_path):
        # DS Config
        print_rank_0("load actor model............")
        model, image_processor, tokenizer = build_model(args=self.args)
        
        if self.args.model_architecture == "default":
            model.load_state_dict(torch.load(os.path.join(actor_path, 'pytorch_model.bin'), map_location='cpu'), strict=False)

        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, self.args.weight_decay, small_lr=self.args.learning_rate_pretraining_components)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.actor_learning_rate, betas=(0.9, 0.95))
        num_update_steps_per_epoch = math.ceil(self.number_dataset / self.args.gradient_accumulation_steps)
        
        if self.args.num_warmup_steps <= 1:
            self.args.num_warmup_steps = int(self.args.num_warmup_steps * self.args.num_train_epochs * num_update_steps_per_epoch)
        else:
            self.args.num_warmup_steps = int(self.args.num_warmup_steps)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.num_train_epochs * num_update_steps_per_epoch,
        )

        ds_config = get_train_ds_config(offload=self.args.offload_actor_model, args=self.args, stage=self.args.actor_zero_stage)
        actor_engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, args=self.args, config=ds_config, lr_scheduler=lr_scheduler, dist_init_required=True)
        
        return actor_engine, image_processor, tokenizer


class RewardEngine():
    def __init__(self, path, writer_config, device):
        self.url = ''
        self.headers = {'Content-Type': 'application/json'}
        self.real_path = path
        self.docker_folder = path
        self.writer_config = writer_config
        self.tool_engine = RestorationToolkit(device=device, score_weight=True)

    def get_reward(self, img_path, tools_text, passcheck=False):
        tools = self.get_tools_from_text(tools_text)
        
        docker_path = os.path.join(self.docker_folder, img_path)
        img_dir = os.path.dirname(img_path)
        output_dir = os.path.join(self.real_path, 'a_tmp_results', os.path.basename(img_dir))
        origin_path = os.path.join(self.real_path, img_path)

        if tools_text == "identity" or passcheck:
            response = self.tool_engine.process_image(tools, origin_path, output_dir, is_identify=True)
        else:
            response = self.tool_engine.process_image(tools, docker_path, output_dir, is_identify=False)
        score = response["score"]
        if not passcheck and (len(tools) == 0 and tools_text != "identity"):
            print(f"debug: get reward tool:{tools}, tools_text:{tools_text}, return all -2 !" )
            return "error!", [-2, -2, -2, -2, -2]
        if response["output_path"] != "error!":
            return "successful!", score
        else:
            return "error!", score

    def get_tools_from_text(self, text):
        pattern = r"\[type:.+?\]:\(model:(.+?)\)"
        return re.findall(pattern, text)

    