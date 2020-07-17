import os
from flask import Flask


app = Flask(__name__)
# Erstelle Secret Key
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
# Lege fest, in welchem Ordner das eingereichte Bild gespeichert wird
UPLOAD_FOLDER = os.path.abspath('shazart\\shazart_web_app\\shazart_app\\static\\uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


from shazart_app import routes