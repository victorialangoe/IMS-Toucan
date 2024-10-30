"""
Example script for fine-tuning the pretrained model to your own data.

Comments in ALL CAPS are instructions
"""

import time

import torch
import wandb

from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    from huggingface_hub import hf_hub_download
    from torch.utils.data import ConcatDataset

    from Modules.ToucanTTS.ToucanTTS import ToucanTTS
    from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
    from Utility.corpus_preparation import prepare_tts_corpus
    from Utility.storage_config import MODELS_DIR
    from Utility.storage_config import PREPROCESSING_DIR

    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    assert gpu_count == 1  # distributed finetuning is not supported

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODELS_DIR, "ToucanTTS_rogaland")  
    os.makedirs(save_dir, exist_ok=True)

    train_data = prepare_tts_corpus(transcript_dict=path_to_transcript_dict(),
                                    corpus_dir=os.path.join(PREPROCESSING_DIR, "rogaland_dialects"),  
                                    lang="nob") 

    model = ToucanTTS()

    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)

    print("Training model")
    train_loop(net=model,
               datasets=[train_data],
               device=device,
               save_directory=save_dir,
               batch_size=16,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS.
               eval_lang="nob",  # THE LANGUAGE YOUR PROGRESS PLOTS WILL BE MADE IN
               warmup_steps=750,
               lr=1e-4,  # if you have enough data (over ~1000 datapoints) you can increase this up to 1e-4 and it will still be stable, but learn quicker.
               path_to_checkpoint=None,
               fine_tune=True if resume_checkpoint is None and not resume else finetune,
               resume=False,
               steps=4000000,
               use_wandb=False,
               train_samplers=[torch.utils.data.RandomSampler(train_data)],
               gpu_count=1)
    if use_wandb:
        wandb.finish()
