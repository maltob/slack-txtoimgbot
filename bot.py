from cmath import log
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
from bot_slack_helper import delete_bot_file, delete_old_files, get_prompts
from bot_config_helper import get_generation_time, get_pipe, get_sd_dimensions,get_num_interations,get_negative_prompt,get_guidance_scale,get_scheduler,get_pipe

from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, KarrasVeScheduler
from torch import torch

load_dotenv()

#locks
generation_lock = Lock()
#Todo change to a locking counter
sd_running_jobs = tsCounter()



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


#Load in config
approved_delete_users = os.environ.get("SLACK_ALLOWED_DELETE").split(",")
img_height= 512
img_width= 512
model_path="CompVis/stable-diffusion-v1-4"
generation_time = 45
guidance_scale = get_guidance_scale(logger,7.5)
num_inference_steps = get_num_interations(logger,50)
negative_prompt = get_negative_prompt(logger,"")

#Environment config

img_height,img_width = get_sd_dimensions(logger,img_height,img_width)

if os.environ.get("SD_MODEL_PATH") and len(os.environ.get("SD_MODEL_PATH")) > 0:
    model_path = os.environ.get("SD_MODEL_PATH")
    logger.debug(f"Set model path to {model_path}")

#Build the StableDiffusionPipeline
pipe = None
scheduler = get_scheduler(logger,DDIMScheduler())
pipe = get_pipe(logger,model_path=model_path)
pipe.enable_attention_slicing()

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)



@app.event("app_mention")
def app_mention(event, say,client):
    txt = event["text"]
    user = event["user"]
    str_txt,neg_txt = get_prompts(logger,txt,negative_prompt)
    channel = event["channel"]
    # Get estimated time to generate the image
    sd_running_jobs.increment()
    time_to_generate = generation_time * (sd_running_jobs.counterValue)
    oMsg = say(f"<@{event['user']}> your photo for \"{str_txt}\" is being created! Give me {time_to_generate} seconds or so to make it.")
    logger.info(f"{str_txt} requested by {user}")
    try:

        #Generate an image and upload it to slack, then delete the info message
        generation_lock.acquire()
        try:
            image = pipe(str_txt, height=img_height,width=img_width,guidance_scale=guidance_scale,negative_prompt=neg_txt,num_inference_steps=num_inference_steps).images[0]
        finally:
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
        logger.info(f"Searching for images to delete at time {ts} due to reaction by {user}")
        delete_bot_file(app,channel=item["channel"],ts=item["ts"],logger=logger)



if os.environ.get("SD_BENCHMARK") and os.environ.get("SD_BENCHMARK").lower()=="true":
    generation_time = get_generation_time(logger=logger,pipe=pipe,img_height=img_height,img_width=img_width,guidance_scale=guidance_scale,negative_prompt=negative_prompt,num_inference_steps=num_inference_steps)
    logger.info(f"Completed will report {generation_time} seconds of gen time")

# Start your app
if __name__ == "__main__":
    #delete_old_files(app,logger)
    logger.info("Starting app")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
