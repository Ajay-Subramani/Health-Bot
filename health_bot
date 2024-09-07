import tkinter as tk
from tkinter import PhotoImage, scrolledtext
import nltk
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure the nltk data path is correct and download required resources
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "What's up"],
        "responses": ["Hi, how can I help you?", "Hello, how can I help you?", "Hey, how can I help you?"]
    },
    {
        "tag": "about",
        "patterns": ["how are you?"],
        "responses": ["I am excellent, thank you. What about you?"]
    },
    {
        "tag": "gratitude",
        "patterns": ["fine", "good", "your working good", "your performing good", "super", "Excellent"],
        "responses": ["Thank you, happy to hear. What's next?"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "name",
        "patterns": ["what is your name", "your name", "who are you", "what's your name", "how can you help me"],
        "responses": ["Hi, I'm Health Bot, your personal assistant. I'm here to help you."]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose", "who created you", "who discovered you", "who developed you", "History", "about you", "what about you", "biography"],
        "responses": ["I am a health bot. My purpose is to assist you. I was developed by an intelligent team from Easwari Engineering College, Chennai. I am designed to assist you and take care of you. How can I help you?"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "heart health",
        "patterns": ["suggest some foods good for heart", "generate a list of food items good for heart", "which foods to consume to maintain a good heart health"],
        "responses": ["Fatty fish (salmon, mackerel)\nNuts and seeds (walnuts, flaxseeds)\nOats and whole grains\nBerries (blueberries, strawberries)\nDark chocolate (in moderation)\nOlive oil"]
    },
    {
        "tag": "diabetes",
        "patterns": ["suggest some foods good for diabetic patients", "which foods to eat for diabetes"],
        "responses": ["Non-starchy vegetables (broccoli, spinach)\nWhole grains (quinoa, brown rice)\nLegumes (beans, lentils)\nLean protein (chicken, fish)\nBerries (blueberries, raspberries)\nNuts and seeds (almonds, chia seeds)"]
    },
    {
        "tag": "digestive health",
        "patterns": ["suggest foods for good digestive health", "which foods to eat for digestive health", "which foods to include in diet for good digestive health"],
        "responses": ["Fiber-rich foods (whole grains, fruits, vegetables)\nYogurt with probiotics\nGinger\nPeppermint\nPapaya\nFennel"]
    },
    {
        "tag": "Bone health",
        "patterns": ["suggest foods for good bone health", "which foods to eat for bone health", "how to get healthy bones", "which foods to include in diet for good bone health"],
        "responses": ["Dairy products (milk, yogurt, cheese)\nLeafy green vegetables (kale, collard greens)\nFatty fish (salmon, sardines)\nFortified foods (orange juice, cereals)\nTofu and soy products"]
    },
    {
        "tag": "Anemia",
        "patterns": ["suggest foods for anemic patients", "which foods to overcome anemia", "which foods to include in diet if one has anemia"],
        "responses": ["Iron-rich foods (red meat, poultry, fish)\nLegumes (beans, lentils)\nDark leafy greens (spinach, kale)\nFortified cereals and bread\nNuts and seeds (pumpkin seeds, cashews)"]
    },
    {
        "tag": "weight management",
        "patterns": ["suggest foods to eat for weight management", "which foods to eat for good weight management"],
        "responses": ["Lean protein sources (chicken, turkey)\nFruits and vegetables\nWhole grains (brown rice, quinoa)\nLegumes (beans, lentils)\nNuts and seeds (in moderation)"]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

def on_send():
    user_input = user_input_entry.get()
    response = chatbot(user_input)
    chatbot_output.insert(tk.END, f"User: {user_input}\nHealth bot: {response}\n\n")
    user_input_entry.delete(0, tk.END)

def on_exit():
    root.destroy()

# GUI setup
root = tk.Tk()
root.iconphoto(False, PhotoImage(file='Tec-robo.png'))  
root.title("Healthbot")
root.config(bg='#333333')

frame = tk.Frame(root, bg="#333333")
frame.pack(pady=5)

chatbot_output = scrolledtext.ScrolledText(frame, width=90, height=40, fg='#ffffff', bg='#262626')
chatbot_output.pack(padx=10)

user_input_entry = tk.Entry(frame, width=100, fg='black', bg='#aaaaaa')
user_input_entry.insert(0, "Ask anything")
user_input_entry.pack(pady=5)

send_button = tk.Button(frame, text="Chat", fg='#ffffff', bg='#242424', command=on_send)
send_button.pack(pady=2)

exit_button = tk.Button(frame, text="Exit", fg='#ffffff', bg='#242424', command=on_exit)
exit_button.pack(pady=2)

root.mainloop()
