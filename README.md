<div align="center" height="130px">
  <h1> Image style transfer Telegram bot </h1>
  <p></p>
</div>

## Table of content
- [About](#about)
- [Getting started](#getting-started)
  - [Techical stack](#tech-stack)
- [Run bot](#run)
  - [Without Docker](#without-docker)
  - [With Docker](#with-docker)
- [Deployment](#deploy)


## 📎 About <a name="about"></a>
This repository is place for image style transfer Telegram bot.
It was developed as a part of Practical Machine Learning and Deep Learning course.

The bot is accessible at [@style_transferer_bot](https://t.me/style_transferer_bot)

## 📌 Getting started <a name="getting-started"></a>

### Technical stack <a name="tech-stack"></a>
The bot is written in **Python** programming language.
It utilizes the fhe [aiogram](https://github.com/aiogram/aiogram) - popular and powerful asynchronous framework
to build bots. In order to store users' state either local memory or [Redis](https://redis.io/) might be used.

## Run bot <a name="run"></a>

### Run locally without Docker <a name="without-docker"></a>

#### Requirements
- In order to launch bot locally, please use Python >= **3.9** and install all the dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- One also needs to install Redis and start Redis server. [Here](https://redis.io/download) you can find the official instructions on how to do it. 
  In order to check that it was installed and is running correctly, type `redis-cli ping`. The word PONG should be displayed.  


- Lastly, run the following commands to download files required for models
  ```bash
  wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat --no-check-certificate -P models/pretrained/
  wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth -P DeepPhotoStyle_pytorch/seg_checkpoint/
  wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth -P DeepPhotoStyle_pytorch/seg_checkpoint/
  ```


#### Start bot
To start the bot, one needs to set up the arguments.
Currently, there are the following arguments needed: `BOT_TOKEN`, `REDIS_ENABLED`, `REDIS_IP`, `REDIS_PORT`, `REDIS_DB`.
The latter four have the default values of `False`, `localhost`, `6379` and `1` correspondingly, so they can be omitted in this section.

To get the description on arguments passing, one can type `python -m bot --help`.

There are several ways how to pass these arguments:

  1) Through the command line  
     Example:  
     ```bash
     python -m bot --bot-token 0123456789:abcdefghijklmnopqrstuvwxyz012345678
     ```

  2) Via config file  
     Create `config.yml` file in the project directory and fill it with data (refer to `config-example.yml` for the example).
     To run, execute the following command: `python -m bot`.

   
### Run using Docker <a name="with-docker"></a>
Firstly, `docker-compose` is required (see the [instructions](https://docs.docker.com/compose/install/)).
Then one needs to run commands that download files required for the models
  ```bash
  wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat --no-check-certificate -P data/files/
  wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth -P data/files/
  wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth -P data/files/
  ```

Lastly, start the container, note that here parameters should be passed via environment variables,
see an example:
```bash
STB_BOT_TOKEN=0123456789:abcdefghijklmnopqrstuvwxyz012345678
STB_REDIS_ENABLED=true
docker-compose up
```
*STB is the prefix for environment variables which means Style Transfer Bot.

## Deployment <a name="deploy"></a>
[immers.cloud](https://immers.cloud/) was used to deploy bot to the server.
