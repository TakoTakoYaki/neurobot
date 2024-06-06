import re
import nltk
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)

API_TOKEN = '7018825493:AAGFaR853kWz6Gni_af1In2S0Gap8x8QVVo'

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Load stop words and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the model
model = load_model('model.keras')

# Load and preprocess datasets
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')

X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text'].values
y_val = df_val['Emotion'].values

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

# Tokenize text data
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

# Text preprocessing functions
def lemmatization(text):
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence

# Emotions and advice dictionary with multiple advices
emotions_advice = {
    'sadness': [
        'Try talking to a friend or watching a good movie.',
        'Take a walk outside and enjoy nature.',
        'Write down your feelings in a journal.'
    ],
    'joy': [
        'Share your happiness with someone you love.',
        'Celebrate your joy with a small treat.',
        'Take some time to appreciate the good things in your life.'
    ],
    'love': [
        'Do something nice for someone you love.',
        'Spend quality time with your loved ones.',
        'Express your feelings openly and honestly.'
    ],
    'fear': [
        'Try to relax with meditation or breathing exercises.',
        'Talk about your fears with a trusted friend.',
        'Engage in a calming activity like reading or listening to music.'
    ],
    'surprise': [
        'Write down your thoughts and feelings in a journal.',
        'Share the surprising news with someone close to you.',
        'Take a moment to reflect on the unexpected event.'
    ],
    'anger': [
        'Engage in physical activity or a hobby to release your anger.',
        'Practice deep breathing exercises.',
        'Write down what made you angry and why.'
    ]
}

# Handler for /start command
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hello! I am a bot that helps identify your emotions and gives advice on what to do. Write me something to get started.")

# Handler for incoming messages
@dp.message_handler()
async def handle_message(message: types.Message):
    sentence = message.text
    sentence = normalized_sentence(sentence)
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=229, truncating='pre')
    prediction = model.predict(padded_sequence)
    result = le.inverse_transform(np.argmax(prediction, axis=-1))[0]
    advice_list = emotions_advice.get(result, ["Sorry, I can't determine your emotion."])
    advice = random.choice(advice_list)
    await message.reply(f'Emotion: {result}\nAdvice: {advice}')

# Start the bot
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
