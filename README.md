# SD - Bot
This bot allows for you to have access to [stable diffusion](https://huggingface.co/blog/stable_diffusion#:~:text=Stable%20Diffusion%20%F0%9F%8E%A8...using%20%F0%9F%A7%A8%20Diffusers%20Stable%20Diffusion%20is,images%20from%20a%20subset%20of%20the%20LAION-5B%20database.) at your fingertips!


## Installation
1. Clone this repository to a folder with at least 5 GB free
```
git clone https://github.com/maltob/slack-txtoimgbot.git
```
2. Open the directory
```
cd slack-txttoimgbot
```
3. Create a venv
```
python -m venv venv
venv/Scripts/activate.ps1
```
4. Install the requirements.txt
```
pip install -r requirements.txt
```
5. Download the stable diffusion repository for the bot
```
huggingface-cli login
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# or if you have less than 8 GB RAM/need FP16 for some other reason

huggingface-cli login
git clone --branch fp16 https://huggingface.co/CompVis/stable-diffusion-v1-4
```
6. Copy example.env to just .env ; then open .env
```
cp example.env .env
notepad .env
```
7. Create the Slack app for your team at https://api.slack.com/apps by importing the [slack_manifest.yml](https://raw.githubusercontent.com/maltob/slack-txtoimgbot/main/slack_manifest.yml)
8. Install to your workspace under OAuth in Slack. Copy the xobb value to set SLACK_BOT_TOKEN= in the .env
9. In Slack App management, open basic information and copy signing secret to the SLACK_SIGNING_SECRET= in .env
9. Further down in basic information generate an App-level token and copy it to the SLACK_APP_TOKEN= in the .env file
9. You can now run the bot. If you get an error, remember to activate the venv with venv/Scripts/activate.ps1
```
python bot.py
```

## .env config options
**SD_IMG_WIDTH**  and **SD_IMG_HEIGHT**

_Set the generated image size. One should be no more than 512 and both should be a multiple of 8_

**SD_PRECISION**

_Set to fp16 for lower precision to save VRAM on cards with less VRAM such as a GTX 1070 or GTX 3060_

**SD_MODEL_PATH**

_Relative path to the downloaded stable diffusion model. There should be a model_index.json inside the folder_ 


**SD_MODEL_AUTH_TOKEN**

_Authentication token to access the model from hugging-face. See [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to generate one_

**SD_BENCHMARK**

_Set to false to not run a generation benchmark when running the bot. It will default to saying give 45 seconds for image to be generated_

**SD_GUIDANCE_SCALE**

_Defaults to 7.5; set higher to have less variance in images/more strongly correlated images_

**SD_ITERATIONS**

_Defaults to 50; set higher for higher quality images with slower generation_

**SD_NEGATIVE_PROMPT**

_Set to words you want to avoid such as low-res_

**SD_SCHEDULER**

_Change the scheduler. Defaults to DDIM. Valid options are **LMS**, **PNDM**, and **KARRASVE**._

**SD_NO_CUDA**

_Set to true to force CPU based inference. This will take tens of minutes per image._