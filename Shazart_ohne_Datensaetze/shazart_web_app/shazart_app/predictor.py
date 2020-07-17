import os
import keras
from keras import backend as K 
import tensorflow as tf
from keras.preprocessing import image 
import numpy as np

def epoch_predictor():
    """ Lädt das Modell und das eingereichte Bild. Erstellt eine Prediction zu dem Bild, 
    wandelt den one hot encodetedn Vektor zum Namen der entsrpechenden Kunstepoche um,
    und gibt diesen zurück.
    """

    CATEGORIES = ["Bauhaus", "Expressionism", "Impressionism", "Renaissance", "Romantic"]

    #Lade Modell und Gewichte
    model = tf.keras.models.load_model(os.path.abspath('shazart\\shazart_web_app\\shazart_app\\multi_epochen.model'))
    model.load_weights(os.path.abspath('shazart\\shazart_web_app\\shazart_app\\multi_epochen_weights.h5'))

    #Lade Bild, konvertiere es zu einem 1D array und erstelle prediction
    img_pred = image.load_img(os.path.abspath('shazart\\shazart_web_app\\shazart_app\\static\\uploads\\file.jpg'), target_size=(224,224))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis = 0)
    result = model.predict(img_pred)

    #Wandle one hot encoding in Namen der Kunstepoche um und gebe diesen zurück
    decoded = tf.argmax(result, axis=1)
    result = CATEGORIES[decoded[0]]
    return result