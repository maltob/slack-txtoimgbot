import os
import logging
import sched
from time import monotonic_ns
from datetime import datetime
from pyexpat import model
from threading import Lock
from tsCounter import tsCounter
# Use the package we installed
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from bot_slack_handler import delete_bot_file, delete_old_files

from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, KarrasVeScheduler
from torch import torch

load_dotenv()

#locks
generation_lock = Lock()
#Todo change to a locking counter
sd_running_jobs = tsCounter()

#Load in config
approved_delete_users = os.environ.get("SLACK_ALLOWED_DELETE").split(",")
img_height= 512
img_width= 512
model_path="CompVis/stable-diffusion-v1-4"
generation_time = 45
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt =""

#setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging_file_handler = logging.FileHandler('app.log')
logging_file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s (%(levelname)s) %(message)s')
logging_file_handler.setFormatter(formatter)
console_log_handler = logging.StreamHandler()
console_log_handler.setLevel(logging.DEBUG)
logger.addHandler(logging_file_handler)
logger.addHandler(console_log_handler)

#Environment config
try:
    if os.environ.get("SD_IMG_HEIGHT") and len(os.environ.get("SD_IMG_HEIGHT")) >0 :
        t_env_height = int(os.environ.get("SD_IMG_HEIGHT"))
        t_env_width = int(os.environ.get("SD_IMG_WIDTH"))
        img_width = t_env_width
        img_height = t_env_height
        logger.debug(f"Loaded height {img_height} and width {img_width} from environment")
except:
    logger.warning(f"Failed to load height and width from environment. Falling back to defaults.")
    print("Error loading the height and width from variables")


if os.environ.get("SD_MODEL_PATH") and len(os.environ.get("SD_MODEL_PATH")) > 0:
    model_path = os.environ.get("SD_MODEL_PATH")
    logger.debug(f"Set model path to {model_path}")

if os.environ.get("SD_NEGATIVE_PROMPT") and len(os.environ.get("SD_NEGATIVE_PROMPT")) > 0:
    negative_prompt = os.environ.get("SD_NEGATIVE_PROMPT")
    logger.debug(f"Set negative prompt to {negative_prompt}")


if os.environ.get("SD_ITERATIONS") and len(os.environ.get("SD_ITERATIONS")) > 0:
    try:
        num_inference_steps = int(os.environ.get("SD_ITERATIONS"))
        logger.debug(f"Set number of inference steps to {num_inference_steps}")
    except:
        logger.debug(f"Failed to parse number of inference steps")

if os.environ.get("SD_GUIDANCE_SCALE") and len(os.environ.get("SD_GUIDANCE_SCALE")) > 0:
    try:
        guidance_scale = float(os.environ.get("SD_GUIDANCE_SCALE"))
        logger.debug(f"Set guidance scale to {guidance_scale}")
    except:
        logger.debug(f"Failed to parse guidance scale")

#Build the StableDiffusionPipeline
pipe = None
scheduler = DDIMScheduler()

if os.environ.get("SD_SCHEDULER") and len(os.environ.get("SD_SCHEDULER")) > 2:
    if os.environ.get("SD_SCHEDULER").upper() == "LMS":
        scheduler = LMSDiscreteScheduler()
        logger.debug(f"Using LMS Scheduler")
    if os.environ.get("SD_SCHEDULER").upper() == "PNDM":
        scheduler = PNDMScheduler()
        logger.debug(f"Using PNDM Scheduler")
    if os.environ.get("SD_SCHEDULER").upper() == "KERRASVE":
        scheduler = KarrasVeScheduler()
        logger.debug(f"Using KerrasVe Scheduler")

if os.environ.get("SD_PRECISION") and len(os.environ.get("SD_PRECISION"))>0 and os.environ.get("SD_PRECISION").lower() == "fp16":
    logger.debug(f"Using fp16 precision")
    if model_path.startswith(".") :
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, revision="fp16")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=os.environ.get("SD_MODEL_AUTH_TOKEN"), torch_dtype=torch.float16, revision="fp16")
else:
    if model_path.startswith(".") :
        pipe = StableDiffusionPipeline.from_pretrained(model_path)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, use_auth_token=os.environ.get("SD_MODEL_AUTH_TOKEN"))

if torch.cuda.is_available() :
    logger.debug(f"Using cuda")
    pipe = pipe.to("cuda")

pipe.enable_attention_slicing()

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)



@app.event("app_mention")
def app_mention(event, say,client):
    txt = event["text"]
    user = event["user"]
    str_txt = txt[(txt.find(" "))+1:]
    neg_txt = negative_prompt
    try :
        indx_neg_prompt = str_txt.index("--")
        neg_txt = str_txt[indx_neg_prompt+2:]
        str_txt = str_txt[:indx_neg_prompt]
        logger.debug(f"Detected negative prompt of {neg_txt} ")
    except:
        try :
            indx_neg_prompt = str_txt.index("-=")
            neg_txt = f"{negative_prompt},{str_txt[indx_neg_prompt+2:]}"
            str_txt = str_txt[:indx_neg_prompt]
            logger.debug(f"Detected negative prompt of {neg_txt} ")
        except:
            logger.debug(f"{str_txt} has no negative prompt, using default of {neg_txt}")
    channel = event["channel"]
    # Get estimated time to generate the image
    sd_running_jobs.increment()
    time_to_generate = generation_time * (sd_running_jobs.counterValue)
    oMsg = say(f"<@{event['user']}> your photo for \"{str_txt}\" is being created! Give me {time_to_generate} seconds or so to make it.")
    logger.info(f"{str_txt} requested by {user}")
    try:

        #Generate an image and upload it to slack, then delete the info message
        generation_lock.acquire()
        image = pipe(txt, height=img_height,width=img_width,guidance_scale=guidance_scale,negative_prompt=neg_txt,num_inference_steps=num_inference_steps).images[0]
        generation_lock.release()
        sd_running_jobs.decrement()
        fp = str_txt.replace(",","_").replace("/","_").replace("\\","_").replace(":","_").replace(".","_")
        file_name =f"files/uf_{user}_{fp}_{datetime.timestamp(datetime.now())}.png"
        image.save(file_name)

        result = client.files_upload(
            channels=channel,
            initial_comment=f"<@{user}> here is your image for \"{str_txt}\"",
            file=file_name,
        )
        logger.info(f"Sent image {file_name} for {str_txt} as requested by {user}")
        #print(result)
        client.chat_delete(token=os.environ.get("SLACK_BOT_TOKEN"),
            channel=oMsg["channel"],
            ts=oMsg["ts"])
    except SlackApiError as e:
        logger.error("Error uploading file: {}".format(e))
        client.chat_update(token=os.environ.get("SLACK_BOT_TOKEN"),
            channel=oMsg["channel"],
            ts=oMsg["ts"],
            text="Ran into an issue uploading your file. Sorry.")
    #print(oMsg)

#reactions are used by some users to delete a file that was uploaded
@app.event("reaction_added")
def handle_reaction_added_events(body):
    if body["event"]["reaction"] == 'x' and body["event"]["user"] in approved_delete_users:
        item = body["event"]["item"]
        ts = item["ts"]
        user = body["event"]["user"]
        #print(item)
        logger.info(f"Searching for images to delete at time {ts} due to reaction by {user}")
        delete_bot_file(app,channel=item["channel"],ts=item["ts"],logger=logger)



if os.environ.get("SD_BENCHMARK") and os.environ.get("SD_BENCHMARK").lower()=="true":
    logger.info("Running benchmark")
    start_ns = monotonic_ns()
    pipe("squid", height=img_height,width=img_width,guidance_scale=guidance_scale,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps,seed=42)
    end_ns = monotonic_ns()
    generation_time = int((end_ns-start_ns)/1_000_000_000) + 5
    logger.info(f"Completed will report {generation_time} seconds of gen time")

# Start your app
if __name__ == "__main__":
    #delete_old_files(app,logger)
    logger.info("Starting app")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
