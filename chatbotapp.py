from flask import Flask, render_template, request, jsonify
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Content from the website (simplified for demonstration)
content = """
Parkinson's disease treatment focuses on managing symptoms and improving quality of life. 
Medications are a primary treatment approach, including Levodopa, Dopamine agonists, MAO-B inhibitors, and COMT inhibitors.
Levodopa is often the most effective medication, but long-term use can lead to dyskinesia.
Dopamine agonists mimic dopamine and can be used alone or with Levodopa.
MAO-B inhibitors help prevent the breakdown of dopamine in the brain.
COMT inhibitors extend the effect of Levodopa.
Physical therapy can improve movement, balance, and coordination.
Occupational therapy helps with daily activities and adapting the home.
Speech therapy addresses speech and swallowing difficulties.
Exercise is crucial for maintaining mobility and flexibility.
Deep brain stimulation (DBS) is a surgical option for advanced Parkinson's.
DBS involves implanting electrodes in the brain to stimulate specific areas.
Lifestyle changes, such as a healthy diet and stress management, are also important.
Support groups can provide emotional support and practical advice.
A balanced diet, including fruits, vegetables, and whole grains, is recommended.
Regular exercise, such as walking, swimming, or cycling, is beneficial.
Stress management techniques, like yoga or meditation, can help.
"""

# Preprocessing the content
sentences = sent_tokenize(content.lower())
words = word_tokenize(content.lower())
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

# Creating a simple response dictionary
response_dict = {
    "medications": [
        "Medications like Levodopa, Dopamine agonists, MAO-B inhibitors, and COMT inhibitors are commonly used.",
        "Levodopa is often the most effective, but discuss potential side effects with your doctor.",
        "Dopamine agonists can be used alone or with Levodopa. Consult your doctor."
    ],
    "physical therapy": [
        "Physical therapy can help improve movement, balance, and coordination.",
        "A physical therapist can create a personalized exercise plan for you."
    ],
    "occupational therapy": [
        "Occupational therapy can help you adapt your home and daily activities.",
        "An occupational therapist can provide strategies for managing daily tasks."
    ],
    "speech therapy": [
        "Speech therapy can address speech and swallowing difficulties.",
        "A speech therapist can help improve communication skills."
    ],
    "exercise": [
        "Regular exercise is crucial for maintaining mobility and flexibility.",
        "Try activities like walking, swimming, or cycling.",
        "Consult your doctor before starting any new exercise program."
    ],
    "dbs": [
        "Deep brain stimulation (DBS) is a surgical option for advanced Parkinson's.",
        "DBS involves implanting electrodes in the brain to stimulate specific areas. Discuss this with your neurologist."
    ],
    "lifestyle": [
        "Lifestyle changes, such as a healthy diet and stress management, are important.",
        "A balanced diet and regular exercise are recommended.",
        "Stress management techniques like yoga or meditation can help."
    ],
    "support groups": [
        "Support groups can provide emotional support and practical advice.",
        "Connecting with others who have Parkinson's can be beneficial."
    ],
    "diet": [
        "A balanced diet with fruits, vegetables, and whole grains is recommended.",
        "Consult a nutritionist for personalized dietary advice."
    ],
    "stress": [
        "Stress management techniques like yoga or meditation can help.",
        "Regular relaxation exercises can reduce stress levels."
    ],
    "treatment": [
        "Parkinson's treatment focuses on managing symptoms and improving quality of life.",
        "A combination of medications, therapies, and lifestyle changes is often used."
    ],
    "symptoms": [
        "This chatbot is for treatment plans only. Please consult a doctor for symptom related questions."
    ]
}

def greetings():
    return "Hello! I'm your Parkinson's Treatment Chatbot. How can I assist you today?"

def thankyou():
    return "You're welcome! If you have any more questions, feel free to ask. Goodbye!"

def chatbot_response(user_input):
    user_input = user_input.lower()
    user_words = word_tokenize(user_input)
    user_filtered_words = [word for word in user_words if word.isalnum() and word not in stop_words]

    # Check for farewell inputs
    if any(word in user_filtered_words for word in ["thank you", "thanks", "bye", "exit", "quit"]):
        return thankyou()

    for keyword, responses in response_dict.items():
        if keyword in user_filtered_words:
            return random.choice(responses)

    for word in user_filtered_words:
        for keyword, responses in response_dict.items():
            if word in keyword:
                return random.choice(responses)

    return "I'm sorry, I don't have information on that. Would you like to know about medications, therapy, exercise, or lifestyle changes?"

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    print(greetings())  # Display greeting when the app starts
    app.run(debug=True)
