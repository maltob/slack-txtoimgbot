import os
from time import monotonic_ns
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, KarrasVeScheduler
from torch import torch

def get_sd_dimensions(logger,default_height,default_width):
    t_env_height = default_height
    t_env_width = default_width
    try:
        if os.environ.get("SD_IMG_HEIGHT") and len(os.environ.get("SD_IMG_HEIGHT")) >0 :
            t_env_height = int(os.environ.get("SD_IMG_HEIGHT"))
            t_env_width = int(os.environ.get("SD_IMG_WIDTH"))
            logger.debug(f"Loaded height {t_env_height} and width {t_env_width} from environment")
    except:
        logger.warning(f"Failed to load height and width from environment. Falling back to defaults of {default_height} {default_width}.")
    return (t_env_height,t_env_width)


def get_num_interations(logger,default_iterations):
    t_iterations = default_iterations
    if os.environ.get("SD_ITERATIONS") and len(os.environ.get("SD_ITERATIONS")) > 0:
        try:
            t_iterations = int(os.environ.get("SD_ITERATIONS"))
            logger.debug(f"Set number of inference steps to {t_iterations}")
        except:
            logger.debug(f"Failed to parse number of inference steps")
    return t_iterations

def get_negative_prompt(logger,default_negative_prompt):
    t_neg_prompt = default_negative_prompt
    if os.environ.get("SD_NEGATIVE_PROMPT") and len(os.environ.get("SD_NEGATIVE_PROMPT")) > 0:
        t_neg_prompt = os.environ.get("SD_NEGATIVE_PROMPT")
        logger.debug(f"Set negative prompt to {t_neg_prompt}")
    return t_neg_prompt

def get_guidance_scale(logger,default_guidance):
    t_guidance = default_guidance
    if os.environ.get("SD_GUIDANCE_SCALE") and len(os.environ.get("SD_GUIDANCE_SCALE")) > 0:
        try:
            t_guidance = float(os.environ.get("SD_GUIDANCE_SCALE"))
            logger.debug(f"Set guidance scale to {t_guidance}")
        except:
            logger.debug(f"Failed to parse guidance scale")
    return t_guidance

def get_scheduler(logger,default_scheduler):
    t_scheduler = default_scheduler
    if os.environ.get("SD_SCHEDULER") and len(os.environ.get("SD_SCHEDULER")) > 2:
        if os.environ.get("SD_SCHEDULER").upper() == "LMS":
            t_scheduler = LMSDiscreteScheduler()
            logger.debug(f"Using LMS Scheduler")
        if os.environ.get("SD_SCHEDULER").upper() == "PNDM":
            t_scheduler = PNDMScheduler()
            logger.debug(f"Using PNDM Scheduler")
        if os.environ.get("SD_SCHEDULER").upper() == "KERRASVE":
            t_scheduler = KarrasVeScheduler()
            logger.debug(f"Using KerrasVe Scheduler")
        if os.environ.get("SD_SCHEDULER").upper() == "DDIM":
            t_scheduler = DDIMScheduler()
            logger.debug(f"Using DDIM Scheduler")
    return t_scheduler

def get_pipe(logger,model_path):
    t_pipe = None
    if os.environ.get("SD_PRECISION") and len(os.environ.get("SD_PRECISION"))>0 and os.environ.get("SD_PRECISION").lower() == "fp16":
        logger.debug(f"Using fp16 precision")
        if model_path.startswith(".") :
            t_pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")
        else:
            t_pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=os.environ.get("SD_MODEL_AUTH_TOKEN"), torch_dtype=torch.float16, revision="fp16")
    else:
        if model_path.startswith(".") :
            t_pipe = StableDiffusionPipeline.from_pretrained(model_path)
        else:
            t_pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=os.environ.get("SD_MODEL_AUTH_TOKEN"))
    #Always add CUDA if available
    if torch.cuda.is_available() :
        logger.debug(f"Using cuda")
        t_pipe = t_pipe.to("cuda")
    return t_pipe

def get_generation_time(logger,pipe,img_height,img_width,guidance_scale,negative_prompt,num_inference_steps,):
    logger.info("Running benchmark")
    start_ns = monotonic_ns()
    pipe("squid", height=img_height,width=img_width,guidance_scale=guidance_scale,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,seed=42)
    end_ns = monotonic_ns()
    return (int((end_ns-start_ns)/1_000_000_000) + 5)
