import os
from shazart_app import app
from flask import render_template, url_for, request, flash
from shazart_app.predictor import epoch_predictor

# In dieser Datei wird das Routing zwischen den einzelnen html templates geregelt

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/uploader', methods = ['POST'])
def upload_file():
    """ Überprüft bei Einreichung eines Bildes, ob es sich um ".jpg" Format handelt 
    falls ja wird das Bild gespeichert und eine prediction zu dem Bild erstellt
    falls nein wird eine Fehlermeldung ausgegeben.
    """
    if request.method == "POST":
        f = request.files['file']
        if f.filename.endswith('.jpg'):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'file.jpg'))
            result = epoch_predictor()
            return render_template("result.html", result = result)
        else:
            flash('Please only upload .jpg files!', 'danger')
            return render_template("home.html", flash = flash)

@app.route('/result', methods=['GET'])
def result():
    return render_template("result.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.after_request
def add_header(r):
    """
    Füge header ein um Caching des eingereichten Bildes zu verhinder, 
    damit bei mehrmaliger Verwendung immer das aktuelle Bild angezeigt wird.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
        


