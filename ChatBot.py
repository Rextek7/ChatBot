# !pip install pymorphy2
import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pymorphy2 import MorphAnalyzer
import pickle
with open('/content/BOT_CONFIG - 22.06.2021.json') as f:
  BOT_CONFIG = json.load(f)

# BOT_CONFIG = {
#     'intents': {
#         'hello': {
#             'examples': ['Привет!', 'Хэлло', 'Хей'],
#             'responses': ['Добрый день', 'Добрый вечер', 'Здравствуйте']
#         },
#         'bye': {
#             'examples': ['Пока...', 'Увидимся!', 'До скорого'],
#             'responses': ['До свиданья', 'До встречи', 'Хорошего дня!']
#         }
#     }
# }
def clean(text):
  cleaned_text = ''
  for char in text.lower():
    if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя ':
      cleaned_text += char #cleaned_text = cleaned_text + char
  return cleaned_text

def get_intent(in_text):
  for intent in BOT_CONFIG['intents'].keys():
    for example in BOT_CONFIG['intents'][intent]['examples']:
      text1 = clean(example)
      text2 = clean(in_text)
      if nltk.edit_distance(text1, text2) / max(len(text1), len(text2)) < 0.4:
        return intent
  return 'Не удалось определить интент'

def bot(in_text):
  intent = get_intent(in_text)

  if intent == 'Не удалось определить интент':
    return 'Извите, я ничего не понял'
  else:
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])

in_text = ''
while in_text != 'Выход':
  in_text = input()
  out_text = bot(in_text)
  print(out_text)

#Обучение Модели
X, y = [], []

for intent in BOT_CONFIG['intents']:
  for example in BOT_CONFIG['intents'][intent]['examples']:
    X.append(example)
    y.append(intent)

morph = MorphAnalyzer()

def lemmatize(text):
  lemmatized_text = []
  for word in text.lower().split():
    lemmatized_text.append(morph.parse(word)[0].normal_form)
  return ' '.join(lemmatized_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = [lemmatize(text) for text in X_train]
X_test = [lemmatize(text) for text in X_test]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))
vectorizer.fit(X_train)
X_vectors = vectorizer.transform(X_train)
len(vectorizer.get_feature_names())

classifier = LogisticRegression(max_iter=200)
classifier.fit(X_vectors, y_train)

classifier.score(X_vectors, y_train)

classifier.score(vectorizer.transform(X_test), y_test)

with open('/content/our_ml_model.pickle', 'wb') as f:
  pickle.dump(classifier, f)

with open('/content/our_vectorizer.pickle', 'wb') as f:
  pickle.dump(vectorizer, f)

with open('/content/our_ml_model.pickle', 'rb') as f:
  loaded_classifier = pickle.load(f)

with open('/content/our_vectorizer.pickle', 'rb') as f:
  loaded_vectorizer = pickle.load(f)


loaded_classifier.predict(loaded_vectorizer.transform(['как дела?']))

#Тестирование
def get_intent_by_ml_model(in_text):
  return classifier.predict(vectorizer.transform([in_text]))[0]

def bot(in_text):
  intent = get_intent_by_ml_model(in_text)

  if intent == 'Не удалось определить интент':
    return 'Извите, я ничего не понял'
  else:
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])

# in_text = ''
# while in_text != 'Выход':
#   in_text = input()
#   out_text = bot(in_text)
#   print(out_text)


#Подключение к Telegram
import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(bot(update.message.text))


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1893870572:AAHYJA6Lhp0iaJo2GQHTqFk_Polas8l0jL8")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

main()