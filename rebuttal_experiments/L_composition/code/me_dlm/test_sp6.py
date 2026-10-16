import os
import re
import math
import time
import json
import torch
import argparse
import subprocess

import torch.distributed as dist

from utils.math_reward import math_reward_fn
from evalplus.data import get_human_eval_plus, get_mbpp_plus

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import AutoTokenizer, AutoModelForCausalLM

TASK = dict()

NFE = {'mask': 0, 'edit': 0}
WALL_T0 = None

def registry(name):

    def _registry(_class):
        TASK[name] = _class
        return _class
    
    return _registry

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def merge_jsonl(pattern, out_path):
    src_path = pattern.replace('*', '[0-7]')
    os.system(f"cat {src_path} > {out_path} && rm {src_path}")

def evalplus(data, path):
    env = os.environ.copy()

    if os.path.exists(path.replace('.jsonl', '_eval_results.json')):
        os.system('rm -rf ' + path.replace('.jsonl', '_eval_results.json'))

    command = ['evalplus.evaluate', '--dataset', data, '--samples', path, '--i_just_wanna_run', 'True']
    result = subprocess.run(command, capture_output=True, env=env)

    scores = re.findall('pass@1:\t([0-9.]*)', result.stdout.decode().strip())

    with open(path.replace('.jsonl', '_eval_results.json'), 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    
    outs = []
    for k, v in data['eval'].items():
        outs.extend(v)

    outs = sorted(outs, key=lambda x: int(x['task_id'].split('/')[1]))
    save_jsonl(outs, path.replace('.jsonl', '_eval_results.jsonl'))

    return {"base_pass@1": float(scores[0]), "base_extra_pass@1": float(scores[1])}

def humaneval(path):
    return evalplus('humaneval', path)

def mbpp(path):
    return evalplus('mbpp', path)

def post_process(code, stops=None):
    try:
        code = code[code.find('```') + 3:]
        code = code[code.find('\n') + 1:]
    except:
        pass

    if stops is None:
        stops = ['\n# Test', '\nif', '\nassert', '\nprint', "\n```", '\ncheck']

    for string in stops:
        if string in code:
            code = code[:code.find(string)]
    
    return code

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

@torch.no_grad()
def mask_diffusion(input_ids, cu_seq_lens, mask_diffusion_step=32, max_prob_score=True):
    answer_mask = input_ids == mask_id

    a = args.init_num // mask_diffusion_step
    b = args.init_num % mask_diffusion_step
    pred_per_step = [a + 1 if i < b else a for i in range(mask_diffusion_step)]

    max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
    for pred_this_step in pred_per_step:
        NFE['mask'] += 1
        outputs = model(
            input_ids=input_ids.unsqueeze(0), 
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_seq_len,
            max_length_k=max_seq_len,
            is_causal=False,
            use_cache=False,
        )

        logit = outputs.logits[0]

        logit[:, del_id] = -torch.inf
        curr_token_ids = torch.argmax(logit, dim=-1)

        if max_prob_score:
            logit[:, pad_id] = -torch.inf
            scores = torch.gather(torch.softmax(logit, dim=-1), dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.rand_like(input_ids, dtype=torch.float)
        
        scores[~answer_mask] = -torch.inf

        for i in range(1, cu_seq_lens.size(0)):
            pred_positions = torch.topk(
                scores[cu_seq_lens[i - 1]:cu_seq_lens[i]], k=pred_this_step
            ).indices + cu_seq_lens[i - 1]

            answer_mask[pred_positions] = False
            input_ids[pred_positions] = curr_token_ids[pred_positions]
    
    return input_ids


@torch.no_grad()
def mask_diffusion_frechet(input_ids, cu_seq_lens, margin, max_prob_score=True):
    """Frechet-profile parallel commit: replaces the fixed per-step quota with
    the profile rule. Sorted confidences c_(1)>=...>=c_(n):
    L_n = sum c_(j) - (n-1); U_n = 1 - c_(n); commit largest n with L_n - U_n > margin.
    Scoring convention identical to the quota arm (softmax after del/pad ban)."""
    answer_mask = input_ids == mask_id
    max_steps = args.init_num
    max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
    steps = 0
    while bool(answer_mask.any()) and steps < max_steps:
        NFE['mask'] += 1
        steps += 1
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_seq_len,
            max_length_k=max_seq_len,
            is_causal=False,
            use_cache=False,
        )
        logit = outputs.logits[0]
        logit[:, del_id] = -torch.inf
        curr_token_ids = torch.argmax(logit, dim=-1)
        if max_prob_score:
            logit[:, pad_id] = -torch.inf
            scores = torch.gather(torch.softmax(logit, dim=-1), dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
        else:
            scores = torch.rand_like(input_ids, dtype=torch.float)
        scores[~answer_mask] = -torch.inf
        for i in range(1, cu_seq_lens.size(0)):
            seg_scores = scores[cu_seq_lens[i - 1]:cu_seq_lens[i]]
            seg_mask = answer_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]]
            m = int(seg_mask.sum().item())
            if m == 0:
                continue
            conf = seg_scores[seg_mask].to(torch.float64)
            vals, _ = torch.sort(conf, descending=True)
            n_idx = torch.arange(m, device=vals.device, dtype=vals.dtype)
            frechet_lb = torch.cumsum(vals, 0) - n_idx
            competitor_ub = 1.0 - vals
            sc = frechet_lb - competitor_ub
            sc[0] = torch.inf
            k = int((sc > margin).sum().clamp(min=1).item())
            pred_positions = torch.topk(seg_scores, k=k).indices + cu_seq_lens[i - 1]
            answer_mask[pred_positions] = False
            input_ids[pred_positions] = curr_token_ids[pred_positions]
    return input_ids

@torch.no_grad()
def edit_diffusion(input_ids, prompt_mask, cu_seq_lens, diffusion_steps=32, pred_per_step=None, max_prob_score=True, max_length=6144):

    real_diffusion_steps = [diffusion_steps] * (cu_seq_lens.size(0) - 1)

    for step in range(diffusion_steps):

        max_seq_len = (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()
        NFE['edit'] += 1
        outputs = model(
            input_ids=input_ids.unsqueeze(0), 
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=max_seq_len,
            max_length_k=max_seq_len,
            is_causal=False,
            use_cache=False,
        )

        curr_token_logits = outputs.logits[0]
        next_token_logits = outputs.next_token_logits[0].roll(1, dims=0)

        curr_token_ids = torch.argmax(curr_token_logits, dim=-1)
        next_token_ids = torch.argmax(next_token_logits, dim=-1)

        # Setting Prompt
        curr_token_ids[prompt_mask.bool()] = input_ids[prompt_mask.bool()]
        next_token_ids[prompt_mask.bool()] = input_ids[prompt_mask.bool()]

        for i in range(1, cu_seq_lens.size(0)):
            if (curr_token_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]] == input_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]]).all() and \
               (next_token_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]] == input_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]]).all() and \
               real_diffusion_steps[i - 1] == diffusion_steps:
                real_diffusion_steps[i - 1] = step + 1

        if pred_per_step is not None:
            if max_prob_score:
                curr_token_score = torch.gather(torch.softmax(outputs.logits[0], dim=-1), dim=-1, index=curr_token_ids.unsqueeze(-1)).squeeze(-1)
                next_token_score = torch.gather(torch.softmax(outputs.next_token_logits[0], dim=-1), dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)
            else:
                curr_token_score = torch.rand_like(input_ids, dtype=torch.float32)
                next_token_score = torch.rand_like(input_ids, dtype=torch.float32)
            
            sub_mask = curr_token_ids != input_ids
            ins_mask = curr_token_ids != next_token_ids

            for i in range(1, cu_seq_lens.size(0)):
                total_sum = sub_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]].sum().item() + ins_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]].sum().item()
                pred_this_step = min(total_sum, pred_per_step)
                
                scores = torch.cat([
                    curr_token_score[cu_seq_lens[i - 1]:cu_seq_lens[i]][sub_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]]],
                    next_token_score[cu_seq_lens[i - 1]:cu_seq_lens[i]][ins_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]]],
                ], dim=0)

                edit_mask = torch.cat([
                    sub_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]],
                    ins_mask[cu_seq_lens[i - 1]:cu_seq_lens[i]],
                ], dim=0)

                scores[~edit_mask] = -torch.inf
                
                pred_positions = torch.topk(scores, k=pred_this_step).indices

                mask = torch.zeros_like(scores).bool()
                mask[pred_positions] = True
                mask = ~(mask & edit_mask)

                sub_change_mask = mask[:input_ids.size(0)]
                ins_change_mask = mask[input_ids.size(0):]
            
                curr_token_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]][sub_change_mask] = input_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]][sub_change_mask]
                next_token_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]][ins_change_mask] = curr_token_ids[cu_seq_lens[i - 1]:cu_seq_lens[i]][ins_change_mask]

        # DEL
        del_mask = torch.ne(curr_token_ids, del_id)
        curr_token_ids = curr_token_ids[del_mask]
        next_token_ids = next_token_ids[del_mask]
        prompt_mask = prompt_mask[del_mask]

        del_mask_int = torch.cat([
            torch.tensor([0], dtype=torch.int, device=del_mask.device),
            torch.cumsum(del_mask, dim=0)
        ], dim=0).int()
        cu_seq_lens = del_mask_int[cu_seq_lens]

        # INS
        save_mask = torch.ne(curr_token_ids, next_token_ids)

        # Avoid exceeding max length
        n = max_length - cu_seq_lens[-1].item()
        idx = save_mask.nonzero(as_tuple=True)[0]
        if idx.numel() > n:
            save_mask[idx[n:]] = False  

        all_mask = torch.stack([save_mask, torch.ones_like(curr_token_ids).bool()], dim=1).view(-1)

        all_prompt_mask = torch.stack([torch.zeros_like(curr_token_ids).bool(), prompt_mask], dim=1).view(-1)
        prompt_mask = all_prompt_mask[all_mask]

        all_mask_int = torch.cat([
            torch.tensor([0], dtype=torch.int, device=all_mask.device),
            torch.cumsum(all_mask, dim=0)
        ], dim=0).int()
        cu_seq_lens = all_mask_int[cu_seq_lens * 2]

        all_input_ids = torch.stack([next_token_ids, curr_token_ids], dim=1).view(-1)
        input_ids = all_input_ids[all_mask]

    return input_ids, cu_seq_lens, prompt_mask, real_diffusion_steps

@torch.no_grad()
def get_answer(string_list):
    mask_texts, edit_texts, inference_time = [], [], []

    input_ids = tokenizer(string_list)['input_ids']
    prompt_mask = []
    for i in range(len(input_ids)):
        prompt_mask.append([1] * len(input_ids[i]) + [0] * args.init_num)
        input_ids[i].extend([mask_id] * args.init_num)
    
    j, total, start_time = 0, len(input_ids), time.time()
    batch_input_ids, batch_prompt_mask, cu_seq_lens = [], [], [0]

    real_diffusion_steps = []
    
    for j in range(total + 1):
        if j == total or len(batch_input_ids) + len(input_ids[j]) > 4096:
            batch_input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long).cuda()
            batch_prompt_mask_tensor = torch.tensor(batch_prompt_mask, dtype=torch.bool).cuda()
            cu_seq_lens_tensor = torch.tensor(cu_seq_lens, dtype=torch.int).cuda()

            local_start_time = time.time()

            if args.frechet_margin is not None:
                batch_input_ids_tensor = mask_diffusion_frechet(batch_input_ids_tensor, cu_seq_lens_tensor, margin=args.frechet_margin)
            else:
                batch_input_ids_tensor = mask_diffusion(batch_input_ids_tensor, cu_seq_lens_tensor, mask_diffusion_step=args.mask_diffusion_step)

            eos_mask = batch_input_ids_tensor != pad_id
            batch_input_ids_tensor = batch_input_ids_tensor[eos_mask]
            batch_prompt_mask_tensor = batch_prompt_mask_tensor[eos_mask]

            eos_mask_int = torch.cat([
                torch.tensor([0], dtype=torch.int, device=eos_mask.device),
                torch.cumsum(eos_mask, dim=0)
            ], dim=0).int()
            cu_seq_lens_tensor = eos_mask_int[cu_seq_lens_tensor]
            if args.edit_diffusion_step > 0:
                batch_input_ids_tensor[cu_seq_lens_tensor[1:] - 1] = eos_id

            for k in range(len(cu_seq_lens_tensor) - 1):
                answer_mask = batch_prompt_mask_tensor[cu_seq_lens_tensor[k]:cu_seq_lens_tensor[k + 1]]
                answer_ids = batch_input_ids_tensor[cu_seq_lens_tensor[k]:cu_seq_lens_tensor[k + 1]]
                answer_ids = answer_ids[~answer_mask]
                mask_texts.append(tokenizer.decode(answer_ids, skip_special_tokens=False))

            batch_input_ids_tensor, cu_seq_lens_tensor, batch_prompt_mask_tensor, _real_diffusion_steps = edit_diffusion(
                batch_input_ids_tensor, batch_prompt_mask_tensor, cu_seq_lens_tensor, 
                diffusion_steps=args.edit_diffusion_step
            )

            real_diffusion_steps.extend(_real_diffusion_steps)

            for k in range(len(cu_seq_lens_tensor) - 1):
                answer_mask = batch_prompt_mask_tensor[cu_seq_lens_tensor[k]:cu_seq_lens_tensor[k + 1]]
                answer_ids = batch_input_ids_tensor[cu_seq_lens_tensor[k]:cu_seq_lens_tensor[k + 1]]
                answer_ids = answer_ids[~answer_mask]
                edit_texts.append(tokenizer.decode(answer_ids, skip_special_tokens=False))
            
            now_time = time.time()
            rest_time = (now_time - start_time) / j * (total - j) / 60
            print(f"Rank {rank} Processing {j}/{total}, Rest Time: {rest_time:.2f} mins")

            _inference_time = (now_time - local_start_time) / (len(cu_seq_lens) - 1)
            inference_time.extend([_inference_time] * (len(cu_seq_lens) - 1))

            batch_input_ids, batch_prompt_mask, cu_seq_lens = [], [], [0]
        
        if j < total:
            batch_input_ids.extend(input_ids[j])
            batch_prompt_mask.extend(prompt_mask[j])
            cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids[j]))

    return mask_texts, edit_texts, real_diffusion_steps, inference_time

@registry('gsm8k')
class GSM8k:

    name = 'gsm8k'

    question_key = 'question'
    answer_key = 'answer'

    data_path = 'data/math/gsm8k.jsonl'

    sample_num = 1

    @classmethod
    def test(cls, result_path):
        samples = load_jsonl(cls.data_path)
        if args.limit > 0:
            samples = samples[:args.limit]

        upsamples = []
        for _ in range(cls.sample_num):
            upsamples.extend(samples)
        
        sample_per_rank = math.ceil(len(upsamples) / dist.get_world_size())
        samples_local = upsamples[rank * sample_per_rank: (rank + 1) * sample_per_rank]

        prompts = []
        for d in samples_local:
            prompts.append(f"<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{d[cls.question_key]}<|im_end|>\n<|im_start|>assistant\n")
        
        mask_preds, edit_preds, real_edit_steps, inference_times = get_answer(prompts)
        
        outs = []
        for sample, mask_pred, edit_pred, real_edit_step, _inference_time in zip(samples_local, mask_preds, edit_preds, real_edit_steps, inference_times):
            mask_correct = math_reward_fn(mask_pred, sample[cls.answer_key])
            edit_correct = math_reward_fn(edit_pred, sample[cls.answer_key])
            outs.append(dict(
                question=sample[cls.question_key],
                answer=sample[cls.answer_key],
                mask_pred=mask_pred,
                edit_pred=edit_pred,
                mask_correct=mask_correct,
                edit_correct=edit_correct,
                real_edit_step=real_edit_step,
                inference_time=_inference_time
            ))

        save_jsonl(outs, os.path.join(result_path, f'{cls.name}_{rank}.jsonl'))
        dist.barrier()
        
        if rank == 0:
            merge_jsonl(os.path.join(result_path, f'{cls.name}_*.jsonl'), os.path.join(result_path, f'{cls.name}.jsonl'))

            mask_right_num, edit_right_num, real_edit_step, inference_time = 0, 0, 0, 0
            data = load_jsonl(os.path.join(result_path, f'{cls.name}.jsonl'))

            for item in data:
                mask_right_num += 1 if item['mask_correct'] else 0
                edit_right_num += 1 if item['edit_correct'] else 0
                real_edit_step += item['real_edit_step']
                inference_time += item['inference_time']

            print(f"{cls.name} Mask Accuracy: {mask_right_num} / {len(data)} = {mask_right_num / len(data):.4f}")
            print(f"{cls.name} Edit Accuracy: {edit_right_num} / {len(data)} = {edit_right_num / len(data):.4f}")
            print(f"{cls.name} Average Real Edit Steps: {real_edit_step / len(data):.2f}")
            
            with open(os.path.join(result_path, f'{cls.name}_accuracy.json'), 'w') as fw:
                json.dump(dict(
                    mask_accuracy=mask_right_num / len(data),
                    edit_accuracy=edit_right_num / len(data),
                    real_edit_steps=real_edit_step / len(data),
                    inference_time=inference_time / len(data)
                ), fw, indent=4)
        
        dist.barrier()

@registry('math500')
class Math500(GSM8k):

    name = 'math500'

    question_key = 'problem'
    answer_key = 'answer'

    data_path = 'data/math/math500.jsonl'

    sample_num = 1

@registry('humaneval')
class Humaneval:

    name = 'humaneval'
    get_score_func = humaneval
    get_dataset_func = get_human_eval_plus

    @classmethod
    def prompt(cls, data):
        return f"<|im_start|>system\nYou are an intelligent programming assistant to produce Python algorithmic solutions.<|im_end|>\n<|im_start|>user\n```python\n{data['prompt'].strip()}\n```<|im_end|>\n<|im_start|>assistant\n"

    @classmethod
    def test(cls, result_path):
        mask_results, edit_results = [], []

        data = load_jsonl(f"data/code/{cls.name}.jsonl")
        
        sample_per_rank = math.ceil(len(data) / dist.get_world_size())
        samples_local = data[rank * sample_per_rank: (rank + 1) * sample_per_rank]

        prompts = []
        for d in samples_local:
            prompts.append(cls.prompt(d))
        
        mask_preds, edit_preds, real_edit_steps, inference_times = get_answer(prompts)

        for d, p, mask_pred, edit_pred, real_edit_step, inference_time in zip(samples_local, prompts, mask_preds, edit_preds, real_edit_steps, inference_times):
            start_coder = p.split('<|im_start|>assistant')[1]

            mask_results.append(dict(task_id=d['task_id'], prompt=d['prompt'], output=mask_pred, completion=post_process(start_coder + mask_pred)))
            edit_results.append(dict(task_id=d['task_id'], prompt=d['prompt'], real_edit_step=real_edit_step, inference_time=inference_time, output=edit_pred, completion=post_process(start_coder + edit_pred)))
        
        save_jsonl(mask_results, os.path.join(result_path, f'{cls.name}_mask_{rank}.jsonl'))
        save_jsonl(edit_results, os.path.join(result_path, f'{cls.name}_edit_{rank}.jsonl'))

        dist.barrier()
        
        if rank == 0:
            merge_jsonl(os.path.join(result_path, f'{cls.name}_mask_*.jsonl'), os.path.join(result_path, f'{cls.name}_mask.jsonl'))
            merge_jsonl(os.path.join(result_path, f'{cls.name}_edit_*.jsonl'), os.path.join(result_path, f'{cls.name}_edit.jsonl'))
            mask_results = cls.get_score_func(os.path.join(result_path, f'{cls.name}_mask.jsonl'))
            edit_results = cls.get_score_func(os.path.join(result_path, f'{cls.name}_edit.jsonl'))

            edit = load_jsonl(os.path.join(result_path, f'{cls.name}_edit.jsonl'))
            real_edit_steps, inference_time = 0, 0
            for item in edit:
                real_edit_steps += item['real_edit_step']
                inference_time += item['inference_time']
            
            print(mask_results, edit_results)
            print(f"Average Real Edit Steps: {real_edit_steps / len(edit):.2f}")

            with open(os.path.join(result_path, f'{cls.name}_accuracy.json'), 'w') as fw:
                json.dump(dict(
                    mask_results=mask_results,
                    edit_results=edit_results,
                    real_edit_steps=real_edit_steps / len(edit),
                    inference_time=inference_time / len(edit)
                ), fw, indent=4)
        
        dist.barrier()

@registry('mbpp')
class MBPP(Humaneval):

    name = 'mbpp'
    get_score_func = mbpp
    get_dataset_func = get_mbpp_plus

    @classmethod
    def prompt(cls, data):
        prompt = data['prompt'].replace('"""', '').strip().split('\n')
        prompt = f"{prompt[0]}\nYour code should satisfy the following assertion:\n```python\n{prompt[1]}\n```"

        return f"<|im_start|>system\nYou are an intelligent programming assistant to produce Python algorithmic solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str)

    parser.add_argument('-n', '--init_num', default=512, type=int)
    parser.add_argument('-m', '--mask_diffusion_step', default=48, type=int)
    parser.add_argument('-e', '--edit_diffusion_step', default=16, type=int)

    parser.add_argument('-t', '--task', default=["gsm8k", "humaneval", "mbpp", "math500"], type=str, nargs='+')
    parser.add_argument('--limit', default=0, type=int, help='use first N samples only (0=all)')
    parser.add_argument('--frechet_margin', default=None, type=float, help='Frechet profile margin delta; None=author quota schedule')
    parser.add_argument('--out_tag', default='', type=str)
    parser.add_argument('--out_root', default=None, type=str, help='results root dir (default: args.path)')

    args = parser.parse_args()

    fm = 'quota' if args.frechet_margin is None else f'frechet{args.frechet_margin}'
    output_name = f"sp6_results-{fm}-n_{args.init_num}-mds_{args.mask_diffusion_step}-eds_{args.edit_diffusion_step}-limit_{args.limit}{args.out_tag}"
    
    out_root = args.out_root or args.path
    os.makedirs(os.path.join(out_root, output_name), exist_ok=True)

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        _attn_implementation="flash_attention_2"
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.path)

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    del_id = tokenizer.convert_tokens_to_ids("<|del|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    vocab_size = len(tokenizer) - len(tokenizer.added_tokens_encoder)

    WALL_T0 = time.time()
    for task in args.task:
        score = TASK[task].test(os.path.join(out_root, output_name))

    wall = time.time() - WALL_T0
    if rank == 0:
        with open(os.path.join(out_root, output_name, 'run_meta.json'), 'w') as fw:
            json.dump(dict(
                arm=('frechet' if args.frechet_margin is not None else 'quota_headon'),
                frechet_margin=args.frechet_margin,
                init_num=args.init_num,
                mask_diffusion_step=args.mask_diffusion_step,
                edit_diffusion_step=args.edit_diffusion_step,
                limit=args.limit,
                tasks=args.task,
                nfe_mask_forwards=NFE['mask'],
                nfe_edit_forwards=NFE['edit'],
                wall_seconds=wall,
            ), fw, indent=4)
        print(f"SP6 RUN META: NFE mask={NFE['mask']} edit={NFE['edit']} wall={wall:.1f}s")