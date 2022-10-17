import os
import logging
from datetime import datetime
# Use the package we installed
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from dotenv import load_dotenv

from diffusers import StableDiffusionPipeline,LMSDiscreteScheduler
from torch import torch

load_dotenv()


approved_delete_users = os.environ.get("SLACK_ALLOWED_DELETE").split(",")

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

logger = logging.getLogger(__name__)

#pipe = StableDiffusionPipeline.from_pretrained("./sd1-4-fp16", torch_dtype=torch.float16, revision="fp16")
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4")
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()


@app.event("app_mention")
def event_test(event, say,client):
    txt = event["text"]
    user = event["user"]
    str_txt = txt[(txt.find(" "))+1:]
    channel = event["channel"]
    oMsg = say(f"<@{event['user']}> your photo for \"{str_txt}\" is being created! Give me 45 seconds or so to make it.")
    try:
        
        image = pipe(txt, height=512,width=512).images[0]
        fp = str_txt.replace(",","_").replace("/","_").replace("\\","_").replace(":","_").replace(".","_")
        file_name =f"files/uf_{user}_{fp}_{datetime.timestamp(datetime.now())}.png"
        image.save(file_name)

        result = client.files_upload(
            channels=channel,
            initial_comment=f"<@{user}> here is your image for \"{str_txt}\"",
            file=file_name,
        )
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
        #print(item)
        delete_bot_file(channel=item["channel"],ts=item["ts"])

def ack_shortcut(ack):
    ack()


def delete_bot_file(channel,ts):
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"),channel=channel,ts_from=(int(float(ts))-1),ts_to=(int(float(ts)))+1)
    myprof =app.client.users_profile_get()
    for file in files["files"]:
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            print("Deleting "+file["name"])
            app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])
        
def delete_old_files():
    myprof =app.client.users_profile_get()
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"))
    for file in files["files"]:
        #print(file)
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        #print(file["name"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            print(file["name"])
            #app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])

# Start your app
if __name__ == "__main__":
    #delete_old_files()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
