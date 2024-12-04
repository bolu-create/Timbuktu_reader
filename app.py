from flask import Flask, render_template, request
import datetime
import re
from google.cloud import vision
from PIL import Image
import cv2
import numpy as np
import os
import vertexai
from vertexai.generative_models import GenerativeModel
import os
from dotenv import load_dotenv
from google.cloud import translate_v2
import pandas as pd
from openpyxl import load_workbook
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import io
import base64
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from pymongo import MongoClient
import requests



def create_app():
    
    app = Flask(__name__)
    client= MongoClient(os.getenv("MONGODB_URI"))#put in your mongodb connection url
    app.db = client.Timbuktu #connect to the database using app.db to save microblog connection inside "app"

    load_dotenv()  # loading all the environmnet variables
    #model = tf.keras.models.load_model(r'models\timbuktu_identifier.h5')
    #my_details = []
    #print("Model loaded:", model)  # Debug print to confirm model loaded


    @app.route("/", methods=["GET","POST"])
    def home(): 
        model = tf.keras.models.load_model(r'models/timbuktu_identifier.h5')
        alert_me = 0
        clear = request.args.get('clear', type=int)
        
        if clear == 1:
            #my_details.clear()
            app.db.djeni.delete_many({})

        
        #Initializing my_details to prevent UnboundlocalError
        my_details=[
                        (
                        content["date"],
                        content["classes"],
                        content["summary"],
                        content["image_url"] 
                        )for content in app.db.djeni.find({})
                    ]
        
        if request.method == "POST":
            my_image = request.files.get("my_image")
            if my_image:
                formatted_date= datetime.datetime.today().strftime("%b %d, %Y")
                image_bytes = my_image.read()
                
                #print(formatted_date)
                ### my_details.append((formatted_date,my_image.filename,my_image.content_type))
                        
                # Preprocess the image (convert to array and scale values)
                #### img = image.load_img(my_image, target_size=(150, 150)) 
                img = image.load_img(io.BytesIO(image_bytes), target_size=(150, 150))       
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalize as done during training
                #img_buff = image.load_img(io.BytesIO(image_bytes))#, target_size=(150, 150))   
                img_buff = image.load_img(io.BytesIO(image_bytes), target_size=(500, 250)) #reducing the image quality      
                

                # Make a prediction
                prediction = model.predict(img_array)
                label = "timbuktu" if prediction[0] > 0.5 else "non_timbuktu"
                print(label)  # Debug the model is working

                if label == "timbuktu":
                    # Google Vision API - Text Detection
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"timbuktu_key.json"
                    
                    def detect_text(image_bytes):
                        client = vision.ImageAnnotatorClient()
                        image = vision.Image(content=image_bytes)  # Image object from byte content
                        response = client.text_detection(image=image)
                        texts = response.text_annotations
                        
                        # Collect OCR results
                        ocr_text = []
                        for text in texts:
                            ocr_text.append(f"\r\n{text.description}")
                        
                        # Handle API errors, if any
                        if response.error.message:
                            raise Exception(
                                "{}\nFor more info on error messages, check: "
                                "https://cloud.google.com/apis/design/errors".format(response.error.message)
                            )
                        
                        return texts[0].description if texts else ""

                    # Call detect_text with the image bytes
                    text_a = detect_text(image_bytes)
                    query = """  \nmake this better, its a result of an OCR on ancient Timbuktu manuscript. Results from OCR's aren't always neat. I want
                                you to return a refined version of the text i.e correct grammatic errors where present and return a better version of the resulting text.
                                I need only the refined text, no other statement"""
                    text_a = text_a + query  # Prepare the refined text with query

                

                    # ............... VERTEX AI...............  

                    # TODO(developer): Update and un-comment below line
                    PROJECT_ID = "fluted-citizen-423010-p4"
                    vertexai.init(project=PROJECT_ID, location="us-central1")
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"timbuktu_key.json"

                    model = GenerativeModel("gemini-1.5-pro-002")

                    def get_vertex_response(question):
                        try:
                            response = model.generate_content(question)
                            return response.text
                        except ValueError:
                            return None

                    # result = get_gemini_response(translated_text)
                    result = get_vertex_response(text_a)

                    # Check if result is None (indicating an error)
                    if result is None:
                        print(f"Skipping image due to response issues.")
                    

                    # ............... TRANSLATION..............

                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"timbuktu_key.json"
                    translate_client = translate_v2.Client()
                    # text = "لونج جونسون، لونج جون جونسون، مواء"
                    text = result
                    target = "en"
                    output = translate_client.translate(text, target_language=target)
                    # print(output['translatedText'])
                    translated_text = output['translatedText']
                    #print(translated_text)
                    
                    # ..........GET A BETTER SUMMARY OF THE TRANSLATED TEXT...........
                    classify_req = """
                    Look at this text, its OCR recovered text from an ancient Manuscript. I know it might not make much sense, but I need you
                    to read it and give me a reasonable summary of what the text is talking about.
                    
                    After your description, can you then return your own version of the text. I know the current version is not coherrent, but 
                    just deliver your own coherrent version of what the text is talking about.
                    
                    Important Note: 
                    1. Only return the summary at all times, no other sentences or words must be added 
                    2. If you don't have a summary, just return "Summary not available" alone!
                    3. The leave some space and return your own version of the text underneath your summary. One white space gap (enter)
                    would be appropriate for this. Remember return only your texts
                    
                    The final format will be:
                    "Summary"
                    *space here*
                    " Your own refined version"
                    """
                    classified_summary = get_vertex_response(translated_text + "\n" + classify_req)

                    # ..........GET CLASSIFICATION OF TEXT...........

                    # Load the tokenizer from Google Drive
                    with open(r'models/tokenizer.pkl', 'rb') as file:
                        tokenizer = pickle.load(file)
                        
                    # Load the saved model
                    model = tf.keras.models.load_model(r'models/djeni_text_classification_model.h5')
                    
                    # Example sentence
                    single_sentence = [translated_text]
                    maxlen=277

                    # Convert to sequence and pad
                    single_sequence = tokenizer.texts_to_sequences(single_sentence)
                    single_padded = pad_sequences(single_sequence, maxlen=maxlen)

                    # Get predictions
                    single_prediction = model.predict(single_padded)

                    # Threshold to binary predictions
                    threshold = 0.5
                    binary_prediction = (single_prediction >= threshold).astype(int)

                    # Map to label names
                    label_names = ["Astronomy", "Agriculture", "Charity", "Doctrine", "Law", "Mathematics", "Medicine", "Theology"]
                    predicted_labels = [label_names[i] for i in range(len(label_names)) if binary_prediction[0][i] == 1]

                    classified_result= ('\n').join(predicted_labels)


                    #......................SETTING UP IMGBB............................
                    # Access the API key
                    IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')

                    if not IMGBB_API_KEY:
                        raise Exception("IMGBB_API_KEY not found. Please check your .env file.")

                    # Use the key in your upload function
                    def upload_to_imgbb(image_bytes):
                        url = "https://api.imgbb.com/1/upload"
                        response = requests.post(url, data={"key": IMGBB_API_KEY}, files={"image": image_bytes})
                        if response.status_code == 200:
                            return response.json()['data']['url']
                        else:
                            raise Exception("Failed to upload image")
                        
                    # In your Flask route
                    with io.BytesIO() as image_buffer:
                        img_buff.save(image_buffer, format="PNG")
                        image_buffer.seek(0)
                        image_url = upload_to_imgbb(image_buffer)
                        app.db.djeni.insert_one({"date": formatted_date, "classes": classified_result, "summary": classified_summary, "image_url": image_url})
                                            
                    #.......................COLLECTING IMAGE...........................
                    # Encode the image to base64 for HTML embedding
                    #buffered = io.BytesIO()
                    #img_buff.save(buffered, format="PNG")
                    #img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    #.................COLLECT AND SAVE INFO INTO MONGODB..................
                    #app.db.djeni.insert_one({"date":formatted_date,"classes":classified_result,"summary":classified_summary,"image":img_base64})
                    

                    #.......................APPEND FINAL DATA FOR LIST..........................
                    #my_details.append((formatted_date,classified_result,translated_text,img_base64))
                    #my_details.append((formatted_date,classified_result,classified_summary,img_base64))
                    
                    #.......................GET DATA FROM DATABASE..........................
                    my_details=[
                        (
                        content["date"],
                        content["classes"],
                        content["summary"],
                        content["image_url"] 
                        )for content in app.db.djeni.find({})
                    ]
                    
                elif label == "non_timbuktu":
                    alert_me = 1
                    
                                
        return render_template("first_page.html", my_details=my_details, alert_me=alert_me)



    @app.route("/details/")
    def details():
        my_details=[
                        (
                        content["date"],
                        content["classes"],
                        content["summary"],
                        content["image_url"] 
                        )for content in app.db.djeni.find({})
                    ]
        index = request.args.get('index', type=int)
        entry = my_details[index] if index is not None and index < len(my_details) else None
        return render_template("second_page.html", entry=entry)
    
    return app
