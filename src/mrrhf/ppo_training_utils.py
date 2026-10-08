import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from utils.data import DST
import re
import copy

def sampling(actor_model,
            img, lang, 
            attention_mask=None,
            pad_token_id=0,
            topk=50,
            topp=0.95,
            do_sample=True,
            max_new_tokens=384,
            num_return_sequences=1,
            temperature=0.75):
    
    generation_kwargs={
        "top_k": topk,
        "top_p": topp,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "temperature": temperature
    }
    max_new_tokens = generation_kwargs["max_new_tokens"]
    generation_kwargs.pop("max_new_tokens")

    batch_size = lang.size()[0]

    all_res = []
 
    actor_model.eval()

    for index in range(batch_size):
        try:
            sub_img = img[index].unsqueeze(0)
        except:
            sub_img = [img[index]]   # reused by the prediction, and there is a situation where the image is None.
        
        sub_attention_mask = attention_mask[index]
        
        sub_lang = lang[index][sum(sub_attention_mask==pad_token_id):].unsqueeze(0)
        res = actor_model.generate(sub_img, sub_lang, 
                                    generation_length=max_new_tokens, 
                                    **generation_kwargs)
        
        all_res.append(res)
    actor_model.train()
    return all_res

def sampling_llava_eval(actor_model,
            img, lang,
            attention_mask=None,
            topk=50,
            topp=0.9,
            do_sample=True,
            max_new_tokens=384,
            num_return_sequences=1,
            temperature=0.5,
            processor=None,
            image_paths=None,
            reward_engine=None,
            saving_folder=None):
    
    generation_kwargs={
        "top_k": topk,
        "top_p": topp,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "temperature": temperature
    }
    max_new_tokens = generation_kwargs["max_new_tokens"]
    generation_kwargs.pop("max_new_tokens")
    batch_size = lang.size()[0]

    actor_model.eval()
    img_score = []
    # batch_infer
    res_all = actor_model.generate(pixel_values=img, input_ids=lang, \
                                     max_new_tokens=max_new_tokens, **generation_kwargs,\
                                     attention_mask=attention_mask, pad_token_id=processor.eos_token_id)
    for index in range(batch_size):
        
        sub_lang = lang[index].unsqueeze(0)
        res = res_all[index][sub_lang.shape[1]:]
        res_text = processor.decode(res)
        res_text = res_text[:res_text.find("<|eot_id|>")]
        img_score.append(reward_engine.get_reward(image_paths[index], res_text)[1])
    return img_score


def sampling_llava_bs(actor_model,
            img, lang,
            attention_mask=None,
            group = 5,
            beam_per_group = 2,
            max_new_tokens=384,
            diversity_penalty=2.0,
            processor=None,
            image_paths=None,
            reward_engine=None,
            reference_tool_map=None):
    
    score_weight_map = {
        'night': [2. / 9, 2. / 9, 0, 2. / 9, 3. / 9],
        'rain_streak': [1. / 5, 1.25 / 5, 1. / 5, 0.75 / 5, 1. / 5],
        'rain_drop': [0, 0.5 / 3, 0, 1.25 / 3, 1.25 / 3],
        'rain_drive': [0.5 / 4, 1.5 / 4, 1. / 4, 1. / 4, 0],
        'snow': [1.5 / 5, 0.75 / 5, 1. / 5, 0.75 / 5, 1. / 5],
        'fog': [1.5 / 5, 0.5 / 5, 1.5 / 5, 0.5 / 5, 1 / 5]
    }
    ALL_TOOLS = [
        "scunet", "restormer", 
        "retinexformer_fivek", "hvicidnet", "lightdiff", 
        "idt", "turbo_rain", "s2former", 
        "ridcp", "kanet", 
        "turbo_snow", "snowmaster", 
        "real_esrgan"
    ]

    generation_config = GenerationConfig(
        num_beam_groups=group,
        diversity_penalty=diversity_penalty,
        num_beams=group*beam_per_group,
        min_length=1,
        num_return_sequences=group
    )
    
    eot_id = processor.encode("<|eot_id|>")[1]
    batch_size = lang.size()[0]

    input_ids = []
    labels = []
    all_score = []
    all_batch_count = []
    
    all_tools_details = []

    actor_model.eval()

    # batch_infer
    res_all = actor_model.generate(pixel_values=img, input_ids=lang, \
                                   max_new_tokens=max_new_tokens, generation_config=generation_config, \
                                    attention_mask=attention_mask, pad_token_id=processor.eos_token_id)
    
    for index in range(batch_size):
        reference_tools = []
        if image_paths[index] in reference_tool_map:
            reference_tools = reference_tool_map[image_paths[index]]
        sub_lang = lang[index].unsqueeze(0)
        res = res_all[index*group:index*group+group][:, sub_lang.shape[1]:]
        for i, tool_text in enumerate(reference_tools):
            reference_tools[i] = normalize_tools(tool_text)
        device = res.device

        for tool in reference_tools:
            encoded_tool = processor.encode(tool)[1:]
            if len(encoded_tool) > res[0].size(0):
                continue
            max_len = res[0].size(0)
            padded_tool = torch.full((max_len,), processor.pad_token_id, dtype=torch.long, device=device)
            padded_tool[:len(encoded_tool)] = torch.tensor(encoded_tool, device=device)
            res = torch.cat((res, padded_tool.unsqueeze(0)))

        query_target = torch.tensor([DST.DEFAULT_LABEL_PADDING_NUM] * (sub_lang[0].shape[0] - 1), device=device).long()
        dummy_target = torch.tensor([DST.DEFAULT_LABEL_PADDING_NUM], device=device).long()

        res_text_set = set()
        img_score = []
        batch_count = 0
        # get identity score
        identity_score = reward_engine.get_reward(image_paths[index], 'identity')[1]
        repeat_penalty = []
        tool_detail = {"identity_score":identity_score}
        for i, r_text in enumerate(res):
            weight_error = None
            r_text = processor.decode(r_text)
            
            if r_text.find("<image>") != -1:
                tool_detail[r_text] = "too many image!!!!"
                print("too many image:", r_text)
                continue

            r_text = r_text[r_text.find("<reason>"):] + processor.eos_token
            r_text = r_text[:r_text.find("<|eot_id|>") + len('<|eot_id|>')]

            if r_text.find("<answer>") == -1:
                print("no answer: ", r_text)
                tool_detail[r_text] = "no answer!!!!" # 0.7
                weight_error = 0.6

            r_text = r_text[r_text.find("<answer>"):]
            temp_tools = get_tools_from_text(r_text)
            intersection = set(temp_tools) & set(ALL_TOOLS)
            if not weight_error and (len(temp_tools) == 0 or intersection != set(temp_tools)):
                print("no tools: ", r_text)
                tool_detail[r_text] = "no tools!!!!"
                weight_error = 0.7

            size = len(res_text_set)
            res_text_set.add(get_tools_from_text(r_text))
            if len(res_text_set) > size:
                res_tmp = res[i]
                mask = res_tmp.eq(eot_id)
                indices = torch.nonzero(mask, as_tuple=False)
                if indices.nelement() > 0:
                    first_match_index = indices[0].item()  
                    res_tmp = res_tmp[:first_match_index + 1]
                    if res_tmp[-1].item() != processor.eos_token_id:
                        res_tmp = torch.cat((res_tmp, torch.tensor([processor.eos_token_id], device=res_tmp.device)))
                
                reward_info = reward_engine.get_reward(image_paths[index], r_text)
                if not weight_error and "error!" == reward_info[0]:
                    tool_detail[r_text] = "no reward!!!!" 
                    weight_error = 0.8
                if weight_error:
                    reward_error = copy.deepcopy(identity_score)
                    reward_error = [weight_error * item for item in reward_error]
                    img_score.append(reward_error)
                else:
                    img_score.append(reward_info[1])
                input_ids.append(torch.cat((sub_lang[0], res_tmp), dim=-1)) 
                labels.append(torch.cat((query_target, res_tmp, dummy_target), dim=0))
                repeat_penalty.append(cal_repeat_penalty(r_text)) 
                batch_count += 1
                # debug
                tool_detail[get_tools_from_text(r_text)] = {"score:": img_score[-1], "repeat_penalty": cal_repeat_penalty(r_text), "error_weight": weight_error if weight_error else 1}


        if batch_count == 0:
            tool_detail["no!!!"] = "big bug!!!"
            print("big bug!!!")

        img_score = torch.tensor(img_score, device=device)
        identity_score = torch.tensor(identity_score, device=device)
        
        # Create a mask for rows that are not all -2
        mask = ~(img_score == -2).all(dim=1)
        img_score_new = torch.zeros(img_score.shape[0], device=device)
        
        score_max = 0
        if mask.any():
            img_score_to_normalize = img_score[mask]
            diff = img_score_to_normalize - identity_score
            # weights for maniqa, musiq, qinstruct
            weights = None
            for name_weight, weight_list in score_weight_map.items():
                if name_weight in image_paths[index]:
                    weights = torch.tensor(weight_list, device=device)
            if weights is None:
                raise ValueError(
                    f"cannot infer degradation type from path {image_paths[index]}; "
                    f"expected the path to contain one of {list(score_weight_map)}")
            weighted_diff = diff * weights
            final_scores = (weighted_diff/identity_score).sum(dim=1)
            img_score_new[mask] = final_scores.float() 
            score_max = torch.max(img_score_new)

        img_score_new[~mask] = -2.0
        for i, rp in enumerate(repeat_penalty): 
            if img_score_new[i] > 0:
                img_score_new[i] /= score_max
                img_score_new[i] -= rp
            img_score_new[i] = max(img_score_new[i], -2)

            
        all_tools_details.append(tool_detail)
        all_batch_count.append(batch_count)
        all_score.extend(img_score_new)
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=processor.pad_token_id
    ) 
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=DST.DEFAULT_LABEL_PADDING_NUM
    )
    actor_model.train()
    scores=torch.tensor(all_score, device=device).float()
    return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(processor.pad_token_id),
            labels=labels, 
            scores=scores, 
            batch_count=all_batch_count,
            tools_details=all_tools_details
        )


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

def compute_logprobs_from_actor_and_ref(actor_model,
                                    images,
                                    input_ids,
                                    input_labels=None,
                                    attention_mask=None,
                                    image_sizes = None,
                                    model_architecture="default"):

    if model_architecture in ["llava", "llava_next"]:
        if image_sizes is not None:
            outputs = actor_model(
                    input_ids=input_ids,
                    pixel_values = images,
                    image_sizes = image_sizes,
                    attention_mask=attention_mask,
                    labels=input_labels,
                    output_hidden_states=True)
            logits = outputs.logits_drop_image
        else:
            outputs = actor_model(
                    input_ids=input_ids,
                    pixel_values = images,
                    attention_mask=attention_mask,
                    labels=input_labels,
                    output_hidden_states=True)
            logits = outputs.logits_drop_image

    logprobs = gather_log_probs(logits[:, :-1, :], input_ids[:, 1:]) 
    ref_logprobs = None 
    
    return logprobs, ref_logprobs


def get_score(logit_label, labels, length_penalty=1.0):
    mask = (labels != DST.DEFAULT_LABEL_PADDING_NUM).float()
    length = mask.sum(-1)
    scores = logit_label.sum(-1) / (length ** length_penalty)
    return scores

def rrhf_loss_func(scores, rw_scores):
    diff = scores.unsqueeze(0) - scores.unsqueeze(-1) # b * b
    rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1) # b * b
    aval = torch.bitwise_and(rw_diff > 0, diff < 0)
    return -(diff*rw_diff)[aval].sum()

def sft_loss_func(logit_label, rw_scores):
    sft_loss = 0
    for i, logit in enumerate(logit_label):
        sft_loss += rw_scores[i] * logit.mean()
    return -sft_loss

def entropy_loss_func(logits, rw_scores):
    weights = torch.exp(-torch.sum(rw_scores, dim=-1))
    probs = torch.softmax(logits, dim=-1)
    return - weights * (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

def get_tools_from_text(text):
    pattern = r"\[type:.+?\]:\(model:(.+?)\)"
    return tuple(re.findall(pattern, text))

def get_text_from_tools(combination):
    ans_list = []
    for item in combination:
        if item in ['scunet', 'restormer']:
            type = "denoise"
        elif item in ['retinexformer_fivek', 'retinexformer_lolv2', 'turbo_night', 'diffll', 'lightdiff']:
            type = "lighten"   
        elif item in ['flare7kpp']:
            type = "deflare"
        elif item in ['real_esrgan']:
            type = "super_resolution"
        elif item in ['idt', 'turbo_rain']:
            type = "derain"
        elif item in ['ridcp']:
            type = "defog"
        elif item in ['turbo_snow']:
            type = "desnow"
        elif item in ['turbo']:
            type = "lighten"
            item = "turbo_night"
        else:
            continue
        ans_list.append(f"[type:{type}]:(model:{item})")
    return ', '.join(ans_list)

def filter_repeat_tools(tools):
    tool_list = [t.strip() for t in tools.split(',')]
    seen = {}
    filtered_tools = []
    for tool in tool_list:
        if not tool: 
            continue
        if 'scunet' in tool:
            if tool not in seen:
                seen[tool] = 1
                filtered_tools.append(tool)
            elif seen[tool] < 2:  # Allow scunet to appear twice
                seen[tool] += 1
                filtered_tools.append(tool)
        else:
            if tool not in seen:  # Other tools appear only once
                seen[tool] = 1
                filtered_tools.append(tool)
    
    # Join filtered tools back together
    return ', '.join(filtered_tools)

def normalize_tools(tools_text):
    return tools_text+'<|eot_id|>'

def cal_repeat_penalty(tools_text):
    tool_dict = {
        "denoise": ["scunet", "restormer"], 
        "lighten": ["retinexformer_fivek", "hvicidnet", "lightdiff"],
        "derain": ["idt", "turbo_rain", "s2former"], 
        "defog":["ridcp", "kanet"], 
        "desnow":["turbo_snow", "snowmaster"], 
        "super_resolution": ["real_esrgan"]
    }
    tools = get_tools_from_text(tools_text)
    if len(tools) == 0:
        return 0
    filtered_tools = set()
    for tool in tools:
        for task, tool_list in tool_dict.items():
            if tool in tool_list:
                filtered_tools.add(task)
                break
    if len(filtered_tools) == 0:
        return 0
    return (len(tools) / len(filtered_tools)) - 1