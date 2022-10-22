
import os
import logging

#Used to clean up all files       
def delete_old_files(app,logger):
    myprof =app.client.users_profile_get()
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"))
    for file in files["files"]:
        #print(file)
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        #print(file["name"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logger.info("File Cleanup - deleting "+file["name"])
            #app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])

#Used to cleanup a file uploaded in a channel at timestamp when requested
def delete_bot_file(app,channel,ts,logger):
    files = app.client.files_list(token=os.environ.get("SLACK_BOT_TOKEN"),channel=channel,ts_from=(int(float(ts))-1),ts_to=(int(float(ts)))+1)
    myprof =app.client.users_profile_get()
    for file in files["files"]:
        user_prof = app.client.users_profile_get(token=os.environ.get("SLACK_BOT_TOKEN"),user=file["user"])
        if file["name"].find("uf_") == 0 and "bot_id" in user_prof["profile"] and user_prof["profile"]["bot_id"] == myprof["profile"]["bot_id"]:
            logger.info("Deleting "+file["name"]+" at user request")
            app.client.files_delete(token=os.environ.get("SLACK_BOT_TOKEN"),file=file["id"])