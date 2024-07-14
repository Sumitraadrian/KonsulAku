import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify, render_template, session
import cv2
import uuid
import numpy as np
import tensorflow as tf
from flask_socketio import SocketIO, emit
from keras.layers import TFSMLayer
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from flask import request, jsonify

# Muat variabel lingkungan dari file .env
load_dotenv()

app = Flask(__name__)
# Ambil variabel lingkungan
SECRET_KEY = os.environ.get('SECRET_KEY', '062502konsulaku')
MODEL_PATH = os.environ.get('MODEL_PATH', 'model/skin_disease_model_saved')

# Atur secret key untuk Flask
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MONGO_URI'] = 'mongodb+srv://sumitraadriansyah:nanang0102*@konsulaku.o95ggwj.mongodb.net/konsulaku?retryWrites=true&w=majority'  # Replace with your MongoDB URI

# Initialize PyMongo
mongo = PyMongo(app)

# Initialize Flask-SocketIO
socketio = SocketIO(app)

# Load model using TFSMLayer
model = tf.keras.Sequential([
    TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
])

# Disease labels and detailed information
disease_info = {
    0: {
        'label': 'Cellulitis',
        'description': 'Cellulitis is a common bacterial skin infection that causes redness, swelling, warmth, and pain in the affected area of the skin. It can occur anywhere on the body but commonly affects the legs.',
        'solution': 'Treatment usually involves oral or intravenous antibiotics, such as cephalexin, dicloxacillin, or clindamycin, to target the bacterial infection. The choice of antibiotic depends on the severity of the infection and the suspected bacteria. In mild cases, oral antibiotics may be sufficient, but severe cases may require hospitalization for intravenous antibiotics. It is crucial to complete the full course of antibiotics as prescribed, even if symptoms improve, to prevent recurrence and antibiotic resistance. Elevating the affected limb can help reduce swelling. Nonsteroidal anti-inflammatory drugs (NSAIDs) like ibuprofen or acetaminophen may be used for pain relief. Proper wound care is essential, including keeping the affected area clean and dry, and covering with sterile dressings. Follow-up care with a healthcare provider is crucial to monitor healing progress, manage any complications such as abscess formation, and adjust treatment if necessary.',
        'symptoms': 'Symptoms of cellulitis include redness, swelling, warmth, pain, and sometimes fever.',
        'causes': 'Cellulitis is caused by bacteria, commonly Staphylococcus or Streptococcus.',
        'risk_factors': 'Risk factors for cellulitis include skin injuries (cuts, burns), a weakened immune system, and obesity.',
        'prevention': 'To prevent cellulitis, keep your skin clean and moisturized, promptly treat cuts and scrapes, and practice good hygiene.',
        'alternative_treatments': 'Alternative treatments for cellulitis may include herbal remedies (consult with a healthcare provider) or complementary therapies.',
        'when_to_see_a_doctor': 'Seek medical attention if you have symptoms of cellulitis, especially if you have a fever or the infection is spreading.'

    },
    1: {
        'label': 'Impetigo',
        'description': 'Impetigo is a highly contagious bacterial skin infection that typically affects children, causing red sores or blisters, often around the nose and mouth. It can also occur on exposed areas of skin.',
        'solution': 'Treatment includes topical antibiotics (such as mupirocin ointment) or oral antibiotics (such as cephalexin or amoxicillin-clavulanate) prescribed by a healthcare provider. Topical antibiotics are applied directly to the sores, while oral antibiotics may be necessary for widespread or severe cases. It is essential to follow the prescribed treatment regimen and complete the full course of antibiotics to prevent recurrence and antibiotic resistance. Keeping the sores clean and covered with sterile gauze or bandages can help prevent the spread of infection to others. Good hygiene practices, such as frequent handwashing with soap and water, can reduce transmission. Avoiding close contact with others until the infection resolves can also prevent spreading the infection.',
        'symptoms': 'Symptoms of impetigo include red sores or blisters, itching, fluid-filled sores, and honey-colored crusts.',
        'causes': 'Impetigo is caused by bacteria, commonly Staphylococcus aureus or Streptococcus pyogenes.',
        'risk_factors': 'Risk factors for impetigo include close contact with infected individuals (especially children), warm and humid environments, and minor skin injuries.',
        'prevention': 'To prevent impetigo, practice good hygiene, keep skin clean and dry, and avoid sharing personal items such as towels and clothing.',
        'alternative_treatments': 'Alternative treatments for impetigo may include home remedies (e.g., vinegar compress) or herbal treatments.',
        'when_to_see_a_doctor': 'Consult a healthcare provider if you suspect impetigo, especially if the infection is spreading or not improving with home care.'

    },
    2: {
        'label': 'Athlete\'s Foot',
        'description': 'Athlete\'s foot, or tinea pedis, is a fungal infection of the skin that commonly affects the feet. It thrives in warm, damp environments such as sweaty socks and shoes.',
        'solution': 'Treatment typically involves topical antifungal creams or ointments (such as terbinafine or clotrimazole) applied directly to the affected area. For more severe or persistent cases, oral antifungal medications (such as fluconazole or itraconazole) may be prescribed. It is important to follow the treatment regimen as prescribed, even if symptoms improve, to ensure complete eradication of the fungus and prevent recurrence. Keeping feet clean and dry, changing socks frequently, and wearing breathable footwear can help prevent recurrence. It\'s crucial to avoid walking barefoot in public areas like locker rooms or showers to reduce the risk of spreading the infection. Regularly disinfecting shoes and socks with antifungal sprays or powders can also help prevent reinfection.',
        'symptoms': 'Symptoms of athlete\'s foot include itching, burning sensation, cracked and flaky skin, and blistering.',
        'causes': 'Athlete\'s foot is caused by fungi, commonly Trichophyton species.',
        'risk_factors': 'Risk factors for athlete\'s foot include wearing tight-fitting shoes, walking barefoot in public areas, and having sweaty feet.',
        'prevention': 'To prevent athlete\'s foot, keep feet clean and dry, wear breathable footwear, and change socks regularly.',
        'alternative_treatments': 'Alternative treatments for athlete\'s foot may include antifungal powders or sprays, and natural remedies such as tea tree oil.',
        'when_to_see_a_doctor': 'See a healthcare provider if you have persistent symptoms of athlete\'s foot, especially if the infection spreads to other parts of the body.'

    },
    3: {
        'label': 'Nail Fungus',
        'description': 'Nail fungus, or onychomycosis, is a fungal infection of the nails that can affect toenails or fingernails. It often begins as a white or yellow spot under the tip of the nail and can cause the nail to thicken, discolor, and crumble at the edges.',
        'solution': 'Treatment may include oral antifungal medications (such as terbinafine or itraconazole) or medicated nail polish (such as ciclopirox). Oral medications are usually necessary for severe or persistent infections. Treatment duration can be prolonged, often requiring several months for complete eradication of the fungus. It\'s important to follow the prescribed treatment regimen consistently to achieve effective results. Keeping nails trimmed and dry can help prevent further infection. Avoiding sharing nail clippers or files and disinfecting nail grooming tools can reduce the risk of spreading the fungus to others or other nails. It\'s essential to wear breathable footwear and avoid tight-fitting shoes to prevent moisture buildup that can promote fungal growth.',
        'symptoms': 'Symptoms of nail fungus include thickened nails, yellow or brown discoloration, brittle or crumbly nails, and distorted nail shape.',
        'causes': 'Nail fungus is caused by fungi, commonly dermatophytes and yeasts.',
        'risk_factors': 'Risk factors for nail fungus include warm and moist environments (e.g., sweaty shoes), nail trauma or injury, and poor circulation.',
        'prevention': 'To prevent nail fungus, keep nails clean and dry, trim nails straight across, and wear moisture-wicking socks and breathable footwear.',
        'alternative_treatments': 'Alternative treatments for nail fungus may include oral antifungal medications and medicated nail polish.',
        'when_to_see_a_doctor': 'Consult a healthcare provider for nail fungus if you notice changes in nail color, texture, or shape, especially if you have diabetes or a weakened immune system.'
    },
    4: {
        'label': 'Ringworm',
        'description': 'Ringworm, or dermatophytosis, is a fungal infection of the skin that typically presents as a circular or ring-shaped rash. Despite its name, it is not caused by a worm but by various types of fungi known as dermatophytes.',
        'solution': 'Treatment usually involves topical antifungal medications (such as clotrimazole cream or terbinafine gel) applied directly to the affected area. For scalp or nail involvement, oral antifungal medications (such as griseofulvin or fluconazole) may be prescribed. It is essential to apply the antifungal treatment as directed and continue treatment for the recommended duration, even if symptoms improve, to ensure complete eradication of the fungus. Keeping the affected area clean and dry can help speed up recovery and prevent spreading to others or other parts of the body. It\'s important to wash hands thoroughly after touching or treating the affected area and to avoid sharing personal items like towels or clothing to prevent reinfection. Regularly disinfecting surfaces and items that come into contact with the infected skin can also help prevent spreading.',
        'symptoms': 'Symptoms of ringworm include a red, scaly rash, circular or ring-shaped patches, itching, and blisters.',
        'causes': 'Ringworm is caused by fungi, commonly Trichophyton, Microsporum, or Epidermophyton species.',
        'risk_factors': 'Risk factors for ringworm include close contact with infected individuals or animals, warm and humid environments, and sharing contaminated items such as towels.',
        'prevention': 'To prevent ringworm, practice good hygiene, avoid sharing personal items, and keep skin dry and clean.',
        'alternative_treatments': 'Alternative treatments for ringworm may include topical antifungal medications and oral antifungal medications for severe cases.',
        'when_to_see_a_doctor': 'Seek medical advice if you suspect ringworm, especially if the rash is widespread, does not improve with over-the-counter treatments, or affects the scalp or nails.'

    },
    5: {
        'label': 'Cutaneous Larva Migrans',
        'description': 'Cutaneous larva migrans is a skin infection caused by hookworm larvae that burrow into the skin, typically resulting in a red, winding rash. It is commonly acquired through contact with contaminated soil or sand in tropical and subtropical regions.',
        'solution': 'Treatment involves topical antiparasitic medications (such as ivermectin cream or albendazole) applied directly to the affected area to kill the larvae and relieve symptoms. It\'s essential to avoid scratching the affected area to prevent bacterial infection. Practicing good hygiene, such as washing hands after outdoor activities and wearing protective footwear in areas where hookworm larvae may be present, can help prevent infection. Avoiding contact with contaminated soil or sand is crucial to reduce the risk of recurrence.',
        'symptoms': 'Symptoms of cutaneous larva migrans include a red, winding rash, itching, and raised tracks under the skin.',
        'causes': 'Cutaneous larva migrans is caused by hookworm larvae, commonly Ancylostoma species and Necator americanus.',
        'risk_factors': 'Risk factors for cutaneous larva migrans include walking barefoot on contaminated soil or sand and engaging in outdoor activities in tropical or subtropical regions.',
        'prevention': 'To prevent cutaneous larva migrans, wear shoes or protective footwear in areas with contaminated soil, and avoid sitting or lying directly on sand or soil.',
        'alternative_treatments': 'Alternative treatments for cutaneous larva migrans may include topical antiparasitic medications and oral antiparasitic medications for severe cases.',
        'when_to_see_a_doctor': 'Consult a healthcare provider if you suspect cutaneous larva migrans, especially if symptoms are severe or do not improve with self-care measures.'

    },
    6: {
        'label': 'Chickenpox',
        'description': 'Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus. It causes an itchy rash with small, fluid-filled blisters that can spread throughout the body.',
        'solution': 'Treatment focuses on relieving symptoms, such as using calamine lotion and antihistamines to reduce itching. Acetaminophen or ibuprofen may be recommended to reduce fever and discomfort. Keeping fingernails short and clean can help prevent scratching and secondary bacterial infections. Vaccination with the chickenpox vaccine (varicella vaccine) is recommended for children and adults who have not had chickenpox to prevent infection. Infected individuals should avoid close contact with others, especially pregnant women, newborns, and individuals with weakened immune systems, until all blisters have crusted over to prevent spreading the virus.',
        'symptoms': 'Symptoms of chickenpox include an itchy rash, blisters, fever, and fatigue.',
        'causes': 'Chickenpox is caused by the varicella-zoster virus.',
        'risk_factors': 'Risk factors for chickenpox include close contact with infected individuals and not being vaccinated against chickenpox.',
        'prevention': 'To prevent chickenpox, vaccination with the chickenpox vaccine (varicella vaccine) is recommended, and avoiding close contact with infected individuals.',
        'alternative_treatments': 'Alternative treatments for chickenpox may include antiviral medications for severe cases.',
        'when_to_see_a_doctor': 'Seek medical attention if you suspect chickenpox, especially if you are at high risk of complications (e.g., adults, pregnant women, individuals with weakened immune systems).'

    },
    7: {
        'label': 'Shingles',
        'description': 'Shingles, or herpes zoster, is a viral infection caused by the reactivation of the varicella-zoster virus, the same virus that causes chickenpox. It typically presents as a painful rash with blisters, usually affecting one side of the body.',
        'solution': 'Treatment includes antiviral medications, such as acyclovir, valacyclovir, or famciclovir, to reduce the severity and duration of symptoms. Pain medications, such as acetaminophen or ibuprofen, and cool compresses can help relieve discomfort. Keeping the rash clean and dry can prevent bacterial infections. Vaccination with the shingles vaccine (Shingrix or Zostavax) is recommended for adults over 50 years old to reduce the risk of developing shingles and postherpetic neuralgia. Individuals with shingles should avoid contact with individuals who have not had chickenpox or the chickenpox vaccine, especially pregnant women, newborns, and individuals with weakened immune systems, until the rash has crusted over.',
        'symptoms': 'Symptoms of shingles include a painful rash, blisters, itching, and sensitivity to touch.',
        'causes': 'Shingles is caused by the reactivation of the varicella-zoster virus.',
        'risk_factors': 'Risk factors for shingles include age (more common in older adults), a weakened immune system, and a previous chickenpox infection.',
        'prevention': 'To prevent shingles, vaccination with the shingles vaccine (Shingrix or Zostavax) is recommended.',
        'alternative_treatments': 'Alternative treatments for shingles may include antiviral medications, pain relievers, and cool compresses.',
        'when_to_see_a_doctor': 'Consult a healthcare provider if you develop a rash that resembles shingles, especially if you are over 50 years old or have a weakened immune system.'

    }
}

# Global variable to store the last predicted disease
last_predicted_disease = None

@app.route('/')
def home():
     # Generate a new session ID for each page load
    session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

# Chatbot introduction message
INTRO_MESSAGE = "Hi! I'm here to help you detect skin diseases from images. Please upload an image of the skin condition you want to diagnose."

# Chatbot closing message
CLOSING_MESSAGE = "Thank you for using our skin disease detection service. Feel free to ask if you have any more questions!"

# Route untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    global last_predicted_disease

    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Simpan file ke direktori static/uploads
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Baca dan preprocess gambar
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi penyakit
    predictions = model.predict(img)

    # Ambil label prediksi
    predicted_class = np.argmax(predictions['dense'][0])
    predicted_label = disease_info[predicted_class]['label']
    predicted_description = disease_info[predicted_class]['description']
    predicted_solution = disease_info[predicted_class]['solution']

    # Simpan prediksi penyakit terakhir
  # Save the last predicted disease
    session['last_predicted_disease'] = disease_info[predicted_class]

    # Return response
    response = {
        'label': predicted_label,
        'description': predicted_description,
        'solution': predicted_solution,
        'filename': file.filename,
        'message': "Do you need any more help or information?"
    }

    # Menambahkan pesan bantuan tambahan setelah deteksi
    response['message'] = "Do you need any more help or information?"
    return jsonify(response)

# Function to save chat history
def save_chat_history(session_id, user_message, bot_response):
    mongo.db.chat_history.update_one(
        {'session_id': session_id},
        {'$push': {'messages': {'user_message': user_message, 'bot_response': bot_response}}},
        upsert=True
    )

# Chatbot endpoint
@socketio.on('message')
def handle_message(message):
    print(f"Received message: {message}")

    session_id = session.get('session_id')
    response = process_message(message)
    save_chat_history(session_id, message, response)
    emit('response', {'response': response})


def process_message(message):
    global last_predicted_disease
    print(f"Received message: {message}")
    
    last_predicted_disease = session.get('last_predicted_disease')
    # Dummy response logic, implement your chatbot logic here
    if 'help' in message.lower() or 'please' in message.lower() and ('detect' in message.lower() or 'skin disease' in message.lower()):
        print("User requests help detecting skin disease")
        return "Okay, please upload an image of the skin condition."

    if 'solution' in message.lower() or 'treatment' in message.lower():
        if last_predicted_disease:
            return f"Solution for {last_predicted_disease['label']}: {last_predicted_disease['solution']}"

    # Check if user asks for disease description
    if 'what is the disease' in message.lower():
        if last_predicted_disease:
            return f"{last_predicted_disease['label']}: {last_predicted_disease['description']}"

    # Provide general information if no specific request found
    for disease_id, info in disease_info.items():
        if info['label'].lower() in message.lower():
            return f"{info['label']}: {info['description']} Solution: {info['solution']}"

        # Check for specific requests
        if any(word in message.lower() for word in ['symptoms', 'causes', 'risk factors', 'prevention', 'alternative treatments', 'when to see a doctor', 'consult a doctor', 'when i must go to doctor']):
            response = ""
            if 'symptoms' in message.lower():
                response = f"Symptoms of {info['label']}: {info['symptoms']}"
            elif 'causes' in message.lower():
                response = f"Causes of {info['label']}: {info['causes']}"
            elif 'risk factors' in message.lower():
                response = f"Risk factors for {info['label']}: {info['risk_factors']}"
            elif 'prevention' in message.lower():
                response = f"Prevention tips for {info['label']}: {info['prevention']}"
            elif 'alternative treatments' in message.lower():
                response = f"Alternative treatments for {info['label']}: {info['alternative_treatments']}"
            elif any(word in message.lower() for word in ['when to see doctor', 'consult a doctor', 'when i must go to doctor']):
                response = f"When to see a doctor for {info['label']}: {info['when_to_see_a_doctor']}"

            if response:
                return response

    # Closing message when no specific request found
    if 'thank you' in message.lower() or 'bye' in message.lower() or 'goodbye' in message.lower():
        return CLOSING_MESSAGE

    # Introduction message at the beginning of the conversation
    if 'hi' in message.lower() or 'hello' in message.lower() or 'start' in message.lower() or 'begin' in message.lower():
        return INTRO_MESSAGE

    return "I'm here to help! You can ask me about the symptoms or treatment of any skin disease listed above."


# Main entry point
if __name__ == '__main__':
    socketio.run(app, debug=True)