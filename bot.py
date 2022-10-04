import telebot
from NN_models.model1 import Model1
from nst_utils import *
from PIL import Image
model1 = Model1()

private_info = open('./private.txt', 'r')
API_TOKEN = private_info.readline().rstrip()
bot = telebot.TeleBot(API_TOKEN)

def download_photo(message):
    raw = message.photo[-1].file_id
    name = "./temp/" + raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name, 'wb') as new_file:
        new_file.write(downloaded_file)
    return name

@bot.message_handler(commands=['generate_image'])
def generate_image(message):
    sent = bot.send_message(message.chat.id, 'Send me style image.')
    bot.register_next_step_handler(sent, get_style_img)

def get_style_img(message):
    sent = bot.send_message(message.chat.id, 'Send me a content image.')
    bot.register_next_step_handler(sent, get_content_img, download_photo(message))

def get_content_img(message, style_img_path):
    content_img_path = download_photo(message)
    r_img = model1.get_res(Image.open(content_img_path), Image.open(style_img_path), learning_rate=2, num_iterations=200, save_amount=20)
    save_image('output/res.png', r_img)
    bot.send_photo(message.chat.id, open('output/res.png', "rb"))

bot.infinity_polling()