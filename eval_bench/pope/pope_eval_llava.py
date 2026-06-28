import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "experiments"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import Conversation, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils import dist_util
from utils.logger import create_logger

from pope_loader import POPEDataSet

torch.multiprocessing.set_sharing_strategy('file_system')

# TODO: statistic
import numpy as np

# TODO: CAI
from utils.llava_utils import set_cai_args, init_cfg_processor
from transformers.generation.logits_process import LogitsProcessorList

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--pope_path", type=str, default="")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--type", type=str, default="random")
    parser.add_argument("--dataset_name", type=str, default="coco")
    # TODO: CAI
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--start_layer", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--use_cai", type=str2bool, default=True)
    parser.add_argument("--use_cd", type=str2bool, default=True)
    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / float(TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred = 0
            pred_list.append(pred)
        else:
            pred = 1
            pred_list.append(pred)
        break
    
    return pred_list, pred


def main():
    args = parse_args()
    dist_util.setup_dist(args)  # Setup DDP
    
    # Setup an experiment folder:
    if dist.get_rank() == 0:
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        print(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Model
    print('Initializing Model')
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )
    
    out_dir = f"{args.out_path}"
    os.makedirs(out_dir, exist_ok=True)

    # TODO: statistic
    per_token_latencies = []
    total_generated_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    # Start Generation
    start_time = time.time()
    
    # TODO: CAI
    set_cai_args(model, args.threshold, args.start_layer, args.use_cai)
    
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"]
        label_list = label_list + list(label)
        
        conv_out = Conversation(
            system="A chat between a curious human and an artificial intelligence assistant. "
                   "The assistant gives helpful, detailed, and polite answers to the human's questions.",
            roles=("USER", "ASSISTANT"),
            version="v1",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        
        qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv_out.append_message(conv_out.roles[0], qu_out)
        conv_out.append_message(conv_out.roles[1], None)
        prompt_out = conv_out.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2
        print("="*100)
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        
        # TODO: CAI
        logits_processor = init_cfg_processor(model, tokenizer, [prompt_out], args.gamma)

        with torch.inference_mode():
            with torch.no_grad():
                torch.cuda.synchronize()    # TODO: statistic
                t0 = time.perf_counter()    # TODO: statistic
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([logits_processor]) if args.use_cd else None   # TODO: CAI
                )
                torch.cuda.synchronize()    # TODO: statistic
                t1 = time.perf_counter()    # TODO: statistic
        
        # TODO: statistic
        num_generated_tokens = output_ids.shape[1] - input_ids.shape[1]
        if num_generated_tokens <= 0:
            num_generated_tokens = 1
        per_token_latencies.append((t1 - t0) * 1000.0 / num_generated_tokens)
        total_generated_tokens += num_generated_tokens
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        pred_list, pred = recorder(outputs, pred_list)
        record = "Yes" if pred == 1 else "No"
        print(f"A: {outputs} ({record})")
        ground_truth = "Yes" if label == 1 else "No"
        print(f"GT: {ground_truth}")

        # dump metric file
        with open(os.path.join(out_dir, f"pope_answers.jsonl"), "a") as f:
            json.dump({"image": image_path,
                       "query": qs,
                       "answer": outputs,
                       "record": record,
                       "ground_truth": ground_truth}, f)
            f.write('\n')
    
    # TODO: statistic
    elapsed_time = time.time() - start_time
    avg_latency_per_token = np.mean(per_token_latencies)
    throughput_tokens = total_generated_tokens / elapsed_time
    peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    if len(pred_list) != 0:
        logger.info(vars(args))
        logger.info(f"Samples: {len(pope_loader)}, Time(s): {elapsed_time:.2f}, Latency(ms/token): {avg_latency_per_token:.2f}, Throughput(tokens/sec): {throughput_tokens:.2f}, GPU memory(MB): {peak_gpu_mem:.2f}")
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )

if __name__ == "__main__":
    main()