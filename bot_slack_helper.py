
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

def get_prompts(logger, txt, default_negative_prompt):
    #Assume @ for the bot is first and chop it off
    t_str_txt = txt[(txt.find(" "))+1:]
    t_neg_txt = default_negative_prompt
    #Check for either of the negative prompt styles and set the prompt
    try :
        indx_neg_prompt = t_str_txt.index("--")
        t_neg_txt = t_str_txt[indx_neg_prompt+2:]
        t_str_txt = t_str_txt[:indx_neg_prompt]
        logger.debug(f"Detected negative prompt of {t_neg_txt} ")
    except:
        try :
            indx_neg_prompt = t_str_txt.index("-=")
            t_neg_txt = f"{default_negative_prompt},{t_str_txt[indx_neg_prompt+2:]}"
            t_str_txt = t_str_txt[:indx_neg_prompt]
            logger.debug(f"Detected negative prompt of {t_neg_txt} ")
        except:
            logger.debug(f"{t_str_txt} has no negative prompt, using default of {t_neg_txt}")
    return t_str_txt,t_neg_txt