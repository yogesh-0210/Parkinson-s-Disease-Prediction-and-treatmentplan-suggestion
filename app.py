from flask import Flask, render_template, request, jsonify
import logging
import joblib
import numpy as np
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
rfc = joblib.load('rf_clf.pkl')

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
    "Parkinson Disease": ["Parkinson's disease (PD) is a neurodegenerative disorder that affects predominately the dopamine-producing (“dopaminergic”) neurons in a specific area of the brain called substantia nigra."],
    "Symptoms":["Symptoms generally develop slowly over years. The progression of symptoms is often a bit different from one person to another due to the diversity of the disease. People with PD may experience:Tremor, mainly at rest and described as pill rolling tremor in hands;/n other forms of tremor are possible/nSlowness and paucity of movement (called bradykinesia and hypokinesia)/nLimb stiffness (rigidity)/nGait and balance problems (postural instability)"],
    "Stages":["Stage 1: During this initial stage, the person has mild symptoms that generally do not interfere with daily activities. Tremor and other movement symptoms occur on one side of the body only. Changes in posture, walking and facial expressions occur.",
    "Stage 2: Symptoms start getting worse. Tremor, rigidity and other movement symptoms affect both sides of the body or the midline (such as the neck and the trunk). Walking problems and poor posture may be apparent. The person is able to live alone, but daily tasks are more difficult and lengthier.",
    "Stage 3: Considered mid-stage, loss of balance (such as unsteadiness as the person turns or when he/she is pushed from standing) is the hallmark. Falls are more common. Motor symptoms continue to worsen. Functionally the person is somewhat restricted in his/her daily activities now, but is still physically capable of leading an independent life. Disability is mild to moderate at this stage.",
    "Stage 4: At this point, symptoms are fully developed and severely disabling. The person is still able to walk and stand without assistance, but may need to ambulate with a cane/walker for safety. The person needs significant help with activities of daily living and is unable to live alone.",
    "Stage 5: This is the most advanced and debilitating stage. Stiffness in the legs may make it impossible to stand or walk. The person is bedridden or confined to a wheelchair unless aided. Around-the-clock care is required for all activities."
    ],
    "Prescription Medications" : ["The choice of medication depends on many variables including your symptoms, other existing health issues (and the medications being used to treat them) and age. Dosages vary greatly depending on a person’s needs and metabolism.",
    "Always remember that medication is only part of the overall treatment plan for combatting PD. Talk to your doctor about available medications, but don’t forget exercise and complementary therapies."
    ],
    "Surgical Treatment Options":["Currently, the two most common surgical treatments available for people living with PD are called deep brain stimulation (DBS) and Duopa™.",
    "Deep Brain Stimulation (DBS):Deep brain stimulation (DBS) is a surgical therapy used to treat certain aspects of Parkinson’s disease (PD). This powerful therapy most addresses the movement symptoms of Parkinson’s and certain side effects caused by medications. DBS may also improve some non-motor symptoms, including sleep, pain, and urinary urgency. It is important to keep in mind that DBS can only help relieve symptoms, not cure or stop disease progression.",
    '''Duopa: Duopa™ therapy is a form of carbidopa-levodopa delivered directly into the intestine in gel form rather than a pill. It is used to treat the same movement symptoms of Parkinson’s disease (PD) that carbidopa-levodopa does, but is designed to improve absorption and reduce “off” times (changes in movement abilities as a levodopa dose wanes) by delivering the drug directly to the small intestine.Before you can start Duopa, a surgery is necessary to make a small hole (called a "stoma") in your abdomen to place a tube in your intestine. A pump then delivers Duopa directly to your intestine through this tube.'''
    ],
    "Physical, Occupational & Speech Therapies":["Physical Therapists help people with PD keep moving well, as long as possible, while enhancing the ability to move. Research shows that physical therapy — including gait and balance training, resistance training and regular exercise — may help improve or hold PD symptoms at bay.",
    "Occupational Therapists help people with PD continue pursuing the activities that make life meaningful and focuses on remaining independent — whether in work, hobbies, social life or in daily activities.",
    "Speech Language Pathologists evaluate speech, voice, communication, swallowing and memory/thinking function. They establish a treatment plan that is consistent with personal goals, such as improving specific communication skills, swallow function and thinking skills."
    ],
    "Over the Counter & Complementary Therapies":["Antioxidants: Vitamin C and E and the Mediterranean Diet",
    "Antioxidants: Vitamin C and E and the Mediterranean Diet"],

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
        "Exercise is an important part of healthy living for everyone. For those with Parkinson’s disease (PD), exercise is more than healthy — it is a vital component to maintaining balance, mobility and activities of daily living. Research shows that exercise and physical activity can not only maintain and improve mobility, flexibility and balance but also ease non-motor PD symptoms such as depression or constipation.",
        "The Parkinson’s Outcomes Project shows that people with PD who start exercising earlier in their disease course for a minimum of 2.5 hours per week experience a slowed decline in quality of life compared to those who start later. Establishing early exercise habits is essential to overall disease management.",
        "To help manage the symptoms of PD, your exercise program should include these key components:/n 1.Aerobic Activity/n 2.Strength Training/n 3.Balance, Agility & Multitasking/n 4.Flexibility"
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

    return "I'm sorry, I don't have information on that. Would you like to know about Parkinson's Diseases medications, therapy, exercise, or lifestyle changes?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parkinson')
def parkinson():
    return render_template('Parkinson.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(feature)) for feature in [
           "MDVP_Fo", "MDVP_Fhi", "MDVP_Flo", "MDVP_Jitter", "MDVP_Jitter_Abs", "MDVP_RAP", "MDVP_PPQ", "Jitter_DDP", "MDVP_Shimmer", "MDVP_Shimmer_dB", "Shimmer_APQ3", "Shimmer_APQ5", "MDVP_APQ", "Shimmer_DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
       ]]

        input_data = np.asarray(features).reshape(1, -1)
        prediction = rfc.predict(input_data)[0]
        result = "Disease Detected" if prediction == 1 else "Normal"
        return render_template('Parkinson.html', result=result)

    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return render_template('Parkinson.html', result=f"Error: {str(e)}")

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')  # New route to serve the chatbot page

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    print(greetings())  # Display greeting when the app starts
    app.run(debug=True)
