import os
import logging
from datetime import datetime
from pyexpat import model
# Use the package we installed
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline,LMSDiscreteScheduler
from torch import torch

load_dotenv()

#Load in config
approved_delete_users = os.environ.get("SLACK_ALLOWED_DELETE").split(",")
img_height= 512
img_width= 512
model_path="CompVis/stable-diffusion-v1-4"


logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.DEBUG)


try:
    if os.environ.get("SD_IMG_HEIGHT") and len(os.environ.get("SD_IMG_HEIGHT")) >0 :
        t_env_height = int(os.environ.get("SD_IMG_HEIGHT"))
        t_env_width = int(os.environ.get("SD_IMG_WIDTH"))
        img_width = t_env_width
        img_height = t_env_height
        logging.debug(f"Loaded height {img_height} and width {img_width} from environment")
except:
    logging.warning(f"Failed to load height and width from environment. Falling back to defaults.")
    print("Error loading the height and width from variables")


if os.environ.get("SD_MODEL_PATH") and len(os.environ.get("SD_MODEL_PATH")) > 0:
    model_path = os.environ.get("SD_MODEL_PATH")
    logging.debug(f"Set model path to {model_path}")

pipe = None
if os.environ.get("SD_PRECISION") and len(os.environ.get("SD_PRECISION"))>0 and os.environ.get("SD_PRECISION").lower() == "fp16":
    logging.debug(f"Using fp16 precision")
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
    logging.debug(f"Using cuda")
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
    oMsg = say(f"<@{event['user']}> your photo for \"{str_txt}\" is being created! Give me 45 seconds or so to make it.")
    try:
        
        image = pipe(txt, height=img_height,width=img_width).images[0]
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

@app.event("reaction_added")
def handle_reaction_added_events(body, logger):
    if body["event"]["reaction"] == 'x' and body["event"]["user"] in approved_delete_users:
        item = body["event"]["item"]
        ts = item["ts"]
        user = body["event"]["user"]
        #print(item)
        logging.info(f"Searching for images to delete at time {ts} due to reaction by {user}")
        delete_bot_file(channel=item["channel"],ts=item["ts"])

def ack_shortcut(ack):
    ack()


def delete_bot_file(channel,ts):
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"),channel=channel,ts_from=(int(float(ts))-1),ts_to=(int(float(ts)))+1)
    myprof =app.client.users_profile_get()
    for file in files["files"]:
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logging.info("Deleting "+file["name"]+" at user request")
            app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])
        
def delete_old_files():
    myprof =app.client.users_profile_get()
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"))
    for file in files["files"]:
        #print(file)
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        #print(file["name"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logging.info("File Cleanup - deleting "+file["name"])
            #app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])

# Start your app
if __name__ == "__main__":
    #delete_old_files()
    logging.info("Starting app")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
