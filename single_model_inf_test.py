import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os
import subprocess
from megatron.global_vars import set_global_variables
from megatron.initialize import initialize_megatron

import torch.nn.functional as F
from torch import nn

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    # with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
    #                          remote_device=None if args.remote_device == 'none' else args.remote_device,
    #                          config_dict_or_path=args.deepspeed_config,
    #                          enabled=args.zero_stage == 3,
    #                          mpu=mpu):
    if args.deepspeed and not args.no_pipeline_parallel:
        model = GPTModelPipe(
            num_tokentypes=0,
            parallel_output=True
        )
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = get_batch_pipe

        # Predompute the attention mask and store it in args. This avoids having to
        # pipeline it as an activation during training. The mask is constant, and thus
        # we can reuse it.
        attention_mask = torch.tril(torch.ones(
            (1, args.seq_length, args.seq_length), device=torch.cuda.current_device())).view(
                1, 1, args.seq_length, args.seq_length)

        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)
        if args.fp16:
            attention_mask = attention_mask.half()
        elif args.bf16:
            attention_mask = attention_mask.bfloat16()

        # Attention mask must be bool.
        args.attn_mask = attention_mask.to(torch.bool)

    else:
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    see_memory_usage(f"After Building Model", force=True)
    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def git_ds_info():
    from deepspeed.env_report import main as ds_report
    ds_report()

    # Write out version/git info
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists('git'):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode('utf-8').strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"
    print(f'**** Git info for Megatron: git_hash={git_hash} git_branch={git_branch} ****')

def command_exists(cmd):
    result = subprocess.Popen(f'type {cmd}', stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0

def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos:
        assert max(args.num_experts) > 1
        loss = loss + moe_loss + mos_loss
        return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}

def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp
    
    if teacher_model:
        with torch.no_grad():
            tea_output, *tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            assert stu_output.size() == tea_output.size(), 'teacher and student output should match in size.'

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.
        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss

def forward_step(data_iterator, model, teacher_model=None):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if args.mos:
        # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
        stu_output, *other_losses = model(tokens, position_ids, attention_mask)
        output_tensor = mpu.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
    else:
        output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                            labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    if args.mos:
        assert model.training
        mos_loss = calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask)
    
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)

if __name__ == "__main__":
    git_ds_info()
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    # set_global_variables(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    args = get_args()

    args.iteration = 0
    # # delayed initialization of DDP-related stuff
    # # We only set basic DDP globals    
    # mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    # # and return function for external DDP manager
    # # to call when it has DDP initialized
    # mpu.set_tensor_model_parallel_rank(args.rank)

    from megatron.training import evaluate_and_print_results, build_train_valid_test_data_iterators, get_model
    
    model = model_provider()

    # print(args)
    ds_engine = deepspeed.init_inference(
        model=model, 
        mp_size = args.tensor_model_parallel_size,
        moe_experts=args.num_experts,
    )

    input_str = "Replace me by any text you'd like."
    from transformers import AutoTokenizer, BertTokenizer 
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # print(encoded_input)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)

    print(ds_engine.module(**encoded_input))
    model = get_model(model_provider)

    # model = ds_engine
    # text_len = len(example_text)

    prefix = 'the end of training for val data'
    iteration = 100

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_datasets_provider)

    evaluate_and_print_results(prefix, forward_step,
                            valid_data_iterator, model,
                            iteration, True)

    # tokenizer = get_tokerizer()

