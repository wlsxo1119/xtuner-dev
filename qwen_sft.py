# Copyright (c) OpenMMLab. All rights reserved.
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import default_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (
    DatasetInfoHook,
    EvaluateChatHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = "./models/qwen2.5-0.5B"  # Path to the model directory
use_varlen_attn = True

# Data
data_path = ["train.jsonl"]
prompt_template = PROMPT_TEMPLATE.mt_default
max_length = 4096
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 10
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 10000
save_total_limit = 10  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 10000
SYSTEM = ''

evaluation_inputs = ['Translate the given English into 한국어. <|English|>: In other words, we shall be considered ordinary from the viewpoint of Korean Buddhists. <|한국어|>:', 'Translate the given 日本語 into 한국어. <|日本語|>: 李林甫は学んだことが見当たらない偉人だったがお世辞という手段で最高の席まで上がった。 <|한국어|>:', "Translate the given English into 한국어. <|English|>: I'm sure the boys were discussing cars. <|한국어|>:", 'Translate the given 한국어 into 中文. <|한국어|>: 리차드 쿠씨는 1954년 일본 출생으로 미국 존스홉킨즈 대학과 대학원에서 경제학을 전공하고 박사학위를 마친 후, 1981년 뉴욕연방은행에 입사하여 조사국과 외국국 등에서 이코노미스트로 활약했다. <|中文|>:', 'Translate the given 한국어 into 中文. <|한국어|>: 또한 지속적인 물가상승으로 인한 실질소득의 감소는 부채가 꾸준히 증가하는 주요한 원인이 되었다. <|中文|>:', 'Translate the given 한국어 into English. <|한국어|>: 근대국가로 거듭나려는 각국의 노력 속에서 문화유산은 국가 정체성을 확립하고 국민을 하나의 공동체로 묶기 위한 주요한 수단이었다. <|English|>:', 'Translate the given 한국어 into English. <|한국어|>: 어떤 시대나 문화에서는 뚱뚱한 것이 부의 상징이자 동경의 대상이다. <|English|>:', 'Translate the given 中文 into 한국어. <|中文|>: 被各种千疮百孔，等待着死去的那一天。 <|한국어|>:', 'Translate the given 中文 into 한국어. <|中文|>: 现有的道德、对市民人性讨论和实践将人性的两大轴心局限在道德、市民人性上，忽视了以“开放的心、好奇心”、批判、省察性思考等为代表的知识人性道德。 <|한국어|>:', 'Translate the given 한국어 into English. <|한국어|>: 인도에선 집에 컴퓨터가 없어도 모바일로 인터넷에 접속할 수 있다. <|English|>:', 'Translate the given English into 한국어. <|English|>: Crisis can be viewed as a historical turning point where the landscape of an entirely new era emerges. <|한국어|>:', 'Translate the given 한국어 into English. <|한국어|>: 1978년 임제종 법맥을 이어받은 성옌 스님은 18년 전 대만 타이베이현에 파구산재단을 창설한 후 신도들의 추앙을 받아왔다. <|English|>:', 'Translate the given 한국어 into 中文. <|한국어|>: 이 글은 이 문제를 좀 더 깊이 다루기 위해 우선 철학적 대화의 전형으로 삼고 있는 소크라테스의 대화를 중심에 두고 이를 철학상담과 관련지어 다루어보고자 한다. <|中文|>:', 'Translate the given 中文 into 한국어. <|中文|>: 但是这些国家是不拥有核武器的零散中立国，在这一点上与朝鲜有争论性的差异。 <|한국어|>:', 'Translate the given 日本語 into 한국어. <|日本語|>: 仏は痛い衆生の病気を直して完快させる医師の中で非常に優れた医師であることを意味する。 <|한국어|>:', 'Translate the given 한국어 into 中文. <|한국어|>: 이는 행정부가 국가안보전략 지침을 명문화된 문서로 작성함으로써 국방 분야에 대한 명확한 지침을 제공하기 위한 목적이었으며, 또한 행정부의 안보전략 수행을 의회가 감시하겠다는 목적도 포함하고 있었다. <|中文|>:', 'Translate the given 한국어 into English. <|한국어|>: 지도층은 없고 지배층만 있는 한국 가톨릭교회는 권위주의의 잘못된 사례 중 하나로 인용할 수 있지 않을까 싶습니다. <|English|>:', 'Translate the given 한국어 into 中文. <|한국어|>: 야당과 보수 성향 국민들의 반대에도 불구하고 국민들의 다수는 지소미아 파기에 찬성하는 경향을 보였다. <|中文|>:', 'Translate the given 中文 into 한국어. <|中文|>: 为了对抗这样的政府政策，从1964年3月开始，以学生及在野党为中心，大规模展开了反对韩日会谈的运动。 <|한국어|>:', 'Translate the given 한국어 into 日本語. <|한국어|>: 무리한 사업 확장으로 서브프라임 사태 이후 신용위기로 큰 타격을 입고도 신속한 대처를 하지 않은 것이 원죄라는 것이다. <|日本語|>:']

train_log_interval = 100
#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
mt_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=default_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn,
)

sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=mt_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template,
    ),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=train_log_interval),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
