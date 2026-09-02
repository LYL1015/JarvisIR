#!/usr/bin/env python
import time
import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
import json
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import SchedulerType

import deepspeed
from rlhf_engine import DeepSpeedRLHFEngine
from ppo_training_utils import compute_logprobs_from_actor_and_ref, sampling_llava_bs, sampling_llava_eval, get_score, rrhf_loss_func, sft_loss_func, entropy_loss_func

import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data import build_dataset, DataCollatorPadToMaxLenForPPOTraining, split_dataset, shuffle_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a multi-modal task")
    parser.add_argument('--data_path',
                        type=str,
                        default='dataset/CleanBench-Real_80k/rrhf_train1.json',
                        help='Where the training data are stored.')
    parser.add_argument('--Offline_data_path',
                        type=str,
                        default='dataset/rrhf_train1_log_check.json',
                        help='Where the Offline training data are stored.')
    parser.add_argument('--data_debug_path',
                        type=str,
                        default=None,
                        help='If provided, will save 10 training samples'
                        'to the path for debug purpose.')

    parser.add_argument('--image_folder',
                type=str,
                default='dataset/CleanBench-Real_80k/images',
                help='Where the image data are stored.')

    parser.add_argument(
        "--data_train_split_ratio",
        type=float,
        default=0.95,
        help="Ratio of dataset to be splitted as train data. The remaining becomes eval data.",
    )
    parser.add_argument('--dataset_names',
                        nargs='*',
                        default=['llava_ppo'],
                        help='Name of training dataset(s) to be used. Accepted format:'
                        '1) a single dataset name, 2) multiple dataset names in the'
                        'form: dataset1 dataset2 ...')
    
    parser.add_argument('--ports', nargs='+', type=int, default=[5000],
                    help='Ports of the tools container.')

    parser.add_argument('--dataset_samples',
                        nargs='*',
                        default=['all'],
                        help='How many samples do we use from each dataset.'
                        'Should be either a integer number or string all which'
                        'means use all samples. For example: all 512 means'
                        'using all samples form first data and 512 samples'
                        'from second data')
    
    parser.add_argument('--dataset_concatenate_samples',
                        nargs='*',
                        default=[1],
                        help='How many samples do we concatenate from each dataset.'
                        'Should be either a integer number or string. 1 which'
                        'means use 1 sample for each datapoint')
    
    parser.add_argument(
        "--max_num_image_per_sample",
        type=int,
        default=8,
        help="The maximum number of images per sample.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="The maximum sequence length, note that image tokens are included.",
    )
    parser.add_argument(
        "--learning_rate_pretraining_components",
        type=float,
        default=0,
        help=
        "Initial learning rate for pre-trained weight, e.g., embedding (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=6,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0.1,
        help="Number of steps (>1) or ratios (<=1) for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='output',
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="llava",
        help=
        "Architecture of pretrained model or model identifier from huggingface.co/models.",
    )
    # deepspeed features
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help=
        "FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models",
    )
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')

    ## from ppo training
    parser.add_argument('--from_checkpoint',
                        type=str,
                        default='checkpoints/huggingface_sft_llava/',
                        help='Path to the trained SFT model.')
    parser.add_argument(
        '--lang_decoder_update',
        action='store_true',
        help='Enable LLM update.')
    # Here we do not set the reference model and critic model.
    # We use the sft model to initialize the reference model, 
    # and use the reward model to initialize the critic model.
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=2,
        help='ZeRO optimization stage for Actor model (and reference).')
    parser.add_argument(
        '--offload_actor_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for actor and reference model.')
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial actor learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        '--max_training_samples_num',
        type=int,
        default=10000,
        help='The maximum number of training samples in the PPO process.')
    
    parser.add_argument(
        '--max_training_step',
        type=int,
        default=2000000,
        help='Maximum training steps for the actor model.')

    parser.add_argument(
        '--save_step',
        type=int,
        default=300,
        help='A checkpoint is saved every specific number of training steps.')

    parser.add_argument(
        '--eval_step',
        type=int,
        default=100,
        help='The evaluation will be conducted every specific number of training steps.')

    parser.add_argument(
        '--max_generation_length_of_sampling',
        type=int,
        default=384,
        help='The max generation langth during sampling.')

    parser.add_argument('--template',
                type=str,
                default='llama_3',
                choices=["default", "llama_2", "llama_3", "llama_3", "vicuna", "llava", "llava_next"],)

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.actor_learning_rate
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"
    return args


def main():
    args = parse_args()

    with open(args.Offline_data_path, 'r') as f:
        lines = json.load(f)
    reference_tool_map = lines
    
    if args.local_rank != -1:
        print("args.local_rank", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    

    args.global_rank = torch.distributed.get_rank()

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    writer_config = SummaryWriter(os.path.join(args.output_dir, str(datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))+f"rank-{args.global_rank}"))
    torch.distributed.barrier()

    tokenizer = None
    reward_tokenizer = None
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.from_checkpoint,
        actor_tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        number_dataset=args.max_training_samples_num,
        args=args,
        writer_config=writer_config)
    
    # Prepare the data
    if len(args.dataset_samples) < len(args.dataset_names):
        assert len(args.dataset_samples) == 1, "when args.dataset_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_samples =  [args.dataset_samples[0]] * len(args.dataset_names)
    if len(args.dataset_concatenate_samples) < len(args.dataset_names):
        assert len(args.dataset_concatenate_samples) == 1, "when args.dataset_concatenate_samples is not the same length as args.dataset_names, it should be only one number"
        args.dataset_concatenate_samples =  [args.dataset_concatenate_samples[0]] * len(args.dataset_names)
    # convert to int
    args.dataset_concatenate_samples = [int(i) for i in args.dataset_concatenate_samples]

    dataset = build_dataset(
        args.data_path,
        args.data_debug_path,
        args.dataset_names,
        args.dataset_samples,
        args.dataset_concatenate_samples,
        args.max_num_image_per_sample,
        vis_processor=rlhf_engine.actor_image_processor,
        vis_root=args.image_folder,
        tokenizer=rlhf_engine.actor_tokenizer_new,
        template=args.template
    )

    # split the dataset into train and evaluation
    np_rng = np.random.RandomState(seed=args.seed)
    dataset = shuffle_dataset(dataset, np_rng)
    train_dataset, eval_dataset = split_dataset(dataset, args.data_train_split_ratio)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLenForPPOTraining(args.max_seq_len, rlhf_engine.actor_tokenizer_new.pad_token_id, rlhf_engine.actor_image_processor.crop_size,),
        generator=torch.Generator(device='cpu')
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=DistributedSampler(eval_dataset, shuffle=True, drop_last=True),
        collate_fn=DataCollatorPadToMaxLenForPPOTraining(args.max_seq_len, rlhf_engine.actor_tokenizer_new.pad_token_id, rlhf_engine.actor_image_processor.crop_size),
        generator=torch.Generator(device='cpu')
    )

    start_epoch = 0

    if args.local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=os.path.join(args.output_dir, 'rrhf_training.log'),
            filemode='w'
        )

    if args.gradient_checkpointing:
        rlhf_engine.actor.gradient_checkpointing_enable()


    def evaluation(eval_dataloader, args, global_step, writer_config, epoch):
        print_rank_0("***** Running evaluation *****", args.global_rank)
        qalign_score = 0
        maniqa_score = 0
        musiq_score = 0
        clipiqa_score = 0
        niqe_score = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = to_device(batch, rlhf_engine.actor.device)
            images = batch["image"].half()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            reward_scores = sampling_llava_eval(rlhf_engine.actor, 
                            images, input_ids,
                            attention_mask=attention_mask, 
                            max_new_tokens=args.max_generation_length_of_sampling, 
                            processor=rlhf_engine.actor_tokenizer_new,
                            image_paths=batch["image_path"], reward_engine=rlhf_engine.reward_engine, saving_folder=f'results/eval_{global_step}')

            qalign_score += sum([score[0] for score in reward_scores])/len(reward_scores)
            maniqa_score += sum([score[1] for score in reward_scores])/len(reward_scores)
            musiq_score += sum([score[2] for score in reward_scores])/len(reward_scores)
            clipiqa_score += sum([score[3] for score in reward_scores])/len(reward_scores)
            niqe_score += sum([score[4] for score in reward_scores])/len(reward_scores)
        del images, input_ids, attention_mask, batch
        torch.cuda.empty_cache()
        qalign_score = qalign_score / len(eval_dataloader)
        maniqa_score = maniqa_score / len(eval_dataloader)
        musiq_score = musiq_score / len(eval_dataloader)
        clipiqa_score = clipiqa_score / len(eval_dataloader)
        niqe_score = niqe_score / len(eval_dataloader)
        print_rank_0(f'Evaluation Reward Score: {qalign_score}, {maniqa_score}, {musiq_score}, {clipiqa_score}, {niqe_score}', args.global_rank)
        # logging 
        if args.global_rank == 0:
            logging.info(f'Evaluation Reward Score: {qalign_score}, {maniqa_score}, {musiq_score}, {clipiqa_score}, {niqe_score}')
        if writer_config and args.global_rank == 0:
            writer_config.add_scalar('qalign_score', qalign_score, step)
            writer_config.add_scalar('maniqa_score', maniqa_score, step)
            writer_config.add_scalar('musiq_score', musiq_score, step)
            writer_config.add_scalar('clipiqa_score', clipiqa_score, step)
            writer_config.add_scalar('niqe_score', niqe_score, step)
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    global_step = 0
    if args.global_rank == 0:
        global_step_global_rank0 = 0
    for epoch in range(start_epoch, args.num_train_epochs):
        print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}", args.global_rank)

        rlhf_engine.actor.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            try:
                batch = to_device(batch, rlhf_engine.actor.device)  #torch.size(1, 3, 224, 224]) #torch.Size([1, 1, 3, 224, 224])
                images = batch["image"].half() 
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                batch_size = input_ids.size()[0]
                image_sizes = None
            
                # Step 1: sampling candidate answers using beam search
                sampling_ans = sampling_llava_bs(rlhf_engine.actor, 
                                images, input_ids,
                                attention_mask=attention_mask, 
                                max_new_tokens=args.max_generation_length_of_sampling, 
                                processor=rlhf_engine.actor_tokenizer_new,
                                group=5, beam_per_group=3, diversity_penalty=2.0,
                                image_paths=batch["image_path"], reward_engine=rlhf_engine.reward_engine,
                                reference_tool_map=reference_tool_map)
                # Step 2: computing reward scores

                # employ reward queue for standardising reward scores.
                reward_scores = sampling_ans["scores"]
                batch_counts = sampling_ans["batch_count"]
                tool_details = sampling_ans["tools_details"]
                if args.global_rank == 0:
                    logging.info(f"here start-----------------------------------epoch{epoch}_step{step}")
                    logging.info(f"batch_counts: {batch_counts}")
                    start = 0
                    for idx in range(batch_size):
                        logging.info(f"batch {idx} reward scores: {reward_scores[start:start+batch_counts[idx]]}")
                        logging.info(f'image_path:{batch["image_path"][idx]}')
                        logging.info(f'debug details:{tool_details[idx]}')
                        start += batch_counts[idx]
                    logging.info('----------------------------------------')
                train_logs = {}
                train_logs["len_reward_score"] = len(reward_scores)
                train_logs["mean_reward_score"] = sum(reward_scores)/len(reward_scores)

                # Step 3: computing KL distance and logprobs
                critic_input_ids = sampling_ans["input_ids"]
                critic_label_ids = sampling_ans["labels"]
                critic_attention_mask = sampling_ans["attention_mask"]
                action_attention_mask = critic_attention_mask[:, 1:]

                all_img = []
                for idx in range(batch_size):
                    # bs, 3, 224, 224
                    all_img.append(images[idx].unsqueeze(0).repeat(batch_counts[idx], 1, 1, 1))
                images = torch.cat(all_img, dim=0)
                #images = images.unsqueeze(1).repeat(1, response_num, 1, 1, 1).view(-1, img_shape[0], img_shape[1], img_shape[2])

                logprobs, ref_logprobs = compute_logprobs_from_actor_and_ref(actor_model=rlhf_engine.actor,
                                        images=images,
                                        input_ids=critic_input_ids,
                                        input_labels=critic_label_ids,
                                        attention_mask=critic_attention_mask,
                                        image_sizes = image_sizes,
                                        model_architecture=args.model_architecture)
                
                # Step 4: compute advantages and returns
                # compute the values
                # run ppo training. 
                # Note that we first implement the case of a minibatch equal to the training batch.
                actor_loss_log = 0
                scores = get_score(logprobs, critic_label_ids)
                success_flag = torch.tensor([1], device=rlhf_engine.actor.device)
                
            except Exception as e:
                logging.info(f"training failed rank {args.global_rank}_epoch{epoch}_step{step}: {str(e)}")
                print(f"training failed rank {args.global_rank}: {str(e)}")
                success_flag = torch.tensor([0], device=rlhf_engine.actor.device)

            dist.all_reduce(success_flag, op=dist.ReduceOp.MIN)
            if success_flag.item() == 0:
                logging.info(f"pass error!!")
                rlhf_engine.actor.zero_grad()
                continue

            rrhf_loss = 0
            sft_loss = 0
            entropy_loss = 0
            # scores/logprobs/reward_scores are flat over all candidates of all samples;
            # slice them per sample using the running offset (same as the logging block above)
            start = 0
            for idx in range(batch_size):
                rrhf_loss += 0.5*rrhf_loss_func(scores[start:start+batch_counts[idx]], reward_scores[start:start+batch_counts[idx]])
                sft_loss += 0.5*sft_loss_func(logprobs[start:start+batch_counts[idx]], reward_scores[start:start+batch_counts[idx]])
                entropy_loss += 0.1*entropy_loss_func(logprobs[start:start+batch_counts[idx]], reward_scores[start:start+batch_counts[idx]])
                start += batch_counts[idx]
            actor_loss = sft_loss + rrhf_loss + entropy_loss
            print_rank_0(f'rrhf_loss: {rrhf_loss}, sft_loss: {sft_loss}')
            train_logs["rrhf_loss"] = rrhf_loss
            train_logs["sft_loss"] = sft_loss
            train_logs["entropy_loss"] = entropy_loss
            
            if args.global_rank == 0:
                logging.info(f'rrhf_loss: {rrhf_loss}, sft_loss: {sft_loss}, entropy_loss: {entropy_loss}')
            rlhf_engine.actor.backward(actor_loss)

            # if accumulation_steps, step actor
            rlhf_engine.actor.step()
            actor_loss_log += actor_loss.detach().clone()
            actor_loss_log = get_all_reduce_mean(actor_loss_log).item()
            train_logs["actor_loss"] = actor_loss_log
            if writer_config and args.global_rank == 0:
                global_step_global_rank0 += 1
                writer_config.add_scalar('Loss/rrhf_loss', train_logs["rrhf_loss"], global_step_global_rank0)
                writer_config.add_scalar('Loss/sft_loss', train_logs["sft_loss"], global_step_global_rank0)
                writer_config.add_scalar('Loss/entropy_loss', train_logs["entropy_loss"], global_step_global_rank0)
                writer_config.add_scalar('Loss/actor_loss', train_logs["actor_loss"], global_step_global_rank0)  

            print_rank_0(
                f'Epoch {epoch+1}, Step: {step+1}, Loss:{actor_loss_log/(global_step+1)}, ', 
                args.global_rank)
            if args.global_rank == 0:
                logging.info(f'Epoch {epoch+1}, Step: {step+1}, Loss:{actor_loss_log/(global_step+1)}')

            global_step += 1
            if global_step % args.save_step == 0:
                model = rlhf_engine.actor
                tokenizer = rlhf_engine.actor_tokenizer_new
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, f'epoch-{epoch}-step-{global_step}')
                if args.global_rank == 0 and args.actor_zero_stage in [1,2]:
                    # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
                    os.makedirs(f'{args.output_dir}/epoch-{epoch}-step-{global_step}', exist_ok=True)
                    WEIGHTS_NAME = "pytorch_model.bin"
                    output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}-step-{global_step}', WEIGHTS_NAME)
                    torch.save(lean_state_dict, output_model_file)
            
            if global_step % args.eval_step == 0:
                evaluation(eval_dataloader, args, global_step, writer_config, epoch)
            del logprobs, ref_logprobs, critic_input_ids, critic_label_ids, critic_attention_mask, action_attention_mask, images, sampling_ans
            torch.cuda.empty_cache()

        evaluation(eval_dataloader, args, global_step, writer_config, epoch)

        model = rlhf_engine.actor
        tokenizer = rlhf_engine.actor_tokenizer_new
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, f'epoch-{epoch}')
        if args.global_rank == 0 and args.actor_zero_stage in [1,2]:
            # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
            model_to_save = model.module if hasattr(model,
                                                    'module') else model
            lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_to_save.state_dict())
            os.makedirs(f'{args.output_dir}/epoch-{epoch}', exist_ok=True)
            WEIGHTS_NAME = "pytorch_model.bin"
            output_model_file = os.path.join(f'{args.output_dir}/epoch-{epoch}', WEIGHTS_NAME)
            torch.save(lean_state_dict, output_model_file)

if __name__ == "__main__":
    main()