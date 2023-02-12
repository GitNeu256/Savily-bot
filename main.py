import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

import telebot
    

API_TOKEN = "5898955683:AAG-VEooW2vIoKV3Diz36Ci1VomR-U-axLQ"

bot = telebot.TeleBot(token = API_TOKEN)


lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents_ru.json", encoding="utf-8").read())
words = pickle.load(open("model/words.pkl", "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))
model = load_model("model/savily_bot.h5")

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in result:
        result_list.append({"intent": classes[r[0]], "probability": str(r[1])})

        return result_list

def get_response(intent_lists, intents_json):
    try:
        tag = intent_lists[0]["intent"]
        list_of_intents = intents_json["intents"]
        result = ""
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    except:
        result = "Извените я вас плохо поняла"
    
    #result = uni2txt(result)
    return result

@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я Савиела.")

@bot.message_handler(func = lambda message: True)
def on_message(message):
    msg = message.text
    ints = predict_class(msg)
    res = get_response(ints, intents)
    bot.send_message(message.chat.id, res)

bot.infinity_polling()
