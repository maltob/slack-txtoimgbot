import os
import logging
from time import monotonic_ns
from datetime import datetime
from pyexpat import model
from threading import Lock
from tsCounter import tsCounter
# Use the package we installed
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline,LMSDiscreteScheduler
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

#Build the StableDiffusionPipeline
pipe = None
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
def event_test(event, say,client):
    txt = event["text"]
    user = event["user"]
    str_txt = txt[(txt.find(" "))+1:]
    channel = event["channel"]
    # Get estimated time to generate the image
    sd_running_jobs.increment()
    time_to_generate = generation_time * (sd_running_jobs.counterValue)
    oMsg = say(f"<@{event['user']}> your photo for \"{str_txt}\" is being created! Give me {time_to_generate} seconds or so to make it.")
    logger.info(f"{str_txt} requested by {user}")
    try:

        #Generate an image and upload it to slack, then delete the info message
        generation_lock.acquire()
        image = pipe(txt, height=img_height,width=img_width).images[0]
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
        delete_bot_file(channel=item["channel"],ts=item["ts"])

def ack_shortcut(ack):
    ack()

#Used to cleanup a file uploaded in a channel at timestamp when requested
def delete_bot_file(channel,ts):
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"),channel=channel,ts_from=(int(float(ts))-1),ts_to=(int(float(ts)))+1)
    myprof =app.client.users_profile_get()
    for file in files["files"]:
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logger.info("Deleting "+file["name"]+" at user request")
            app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])

#Used to clean up all files       
def delete_old_files():
    myprof =app.client.users_profile_get()
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"))
    for file in files["files"]:
        #print(file)
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        #print(file["name"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logger.info("File Cleanup - deleting "+file["name"])
            #app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])


if os.environ.get("SD_BENCHMARK") and os.environ.get("SD_BENCHMARK").lower()=="true":
    logger.info("Running benchmark")
    start_ns = monotonic_ns()
    pipe("squid", height=img_height,width=img_width)
    end_ns = monotonic_ns()
    generation_time = int((end_ns-start_ns)/1_000_000_000) + 5
    logger.info(f"Completed will report {generation_time} seconds of gen time")

# Start your app
if __name__ == "__main__":
    #delete_old_files()
    logger.info("Starting app")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
