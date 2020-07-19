# Shazart - Data Exploration Project
Teilnehmer: Florian Gemmer, Silvan Biewald, Klemens Gerber und Arnold Hajnal

# Zielsetzung des Projekts
Shazart soll eine Plattform für kunstinteressierte Nutzer sein, die gerne einen Anhaltspunkt für ihre Recherche zu Gemälde haben wollen, über die sie ohne weiteres keine genaueren Informationen bekommen können. Ausserdem soll Shazart, Künstlerinnen und Künstlern eine Möglichkeit geben ihre eigenen Werke hinsichtlich Ähnlichkeiten zu klassischen Kunstepochen zu untersuchen, indem sie von unserem System klassifiziert werden. Um dieses Ziel zu erreichen wurde ein CNN trainiert, fünf klassische Kunstepochen zu unterscheiden und ähnlichkeiten zu klassischen Kunstwerken zu erkennen.

# Hinweise zur Struktur des Projekts

Dieses Github Repository beinhaltet alle Infos, die Sie zum ausführen unseres Codes brauchen. 
Hier finden Sie:
  - Unser finales, trainiertes Modell
  - Den Code, den wir zum trainieren benutzt haben (Als Juypter Notebook)
  - Unsere Abschlusspräsentation als Google Sheet 
  - Unseren Projektreport für tiefer gehende Informationen
  - Einen Link zu unserem gesamten Datensatz auf Google Drive (Link: Placeholder)

Zum ausführen des Modells, müssen Sie lediglich die entsprechende Python Datei "Run.py" im "shazart_web_app" Ordner ausführen. Diesen finden Sie im Shazart_ohne_Datensaetze" Ordner 
Es wird sich eine Webseite im Browser öffnen, auf der Sie ein beliebiges Bild aus den von uns behandelten Epochen Kubismus, Impressionismus, Expressionismus
Romantik oder Barock hochladen können. Unser Modell wird ihnen dann die entsprechende Prediction ausgeben.

Der Quellcode:
Wir haben Jupyter Notebook als IDE verwendet, deshalb sollten Sie das für die beste Erfahrung auch tun. 
Nötige dependencies sind:
  - Matplotlib/Pyplot
  - Keras
  - Tensorflow
  - os
  - Numpy
  - Itertools
  - scikit-learn
  
  Den Quellcode finden Sie unter: DataExplorationProject/Shazart_ohne_Datensaetze/image_classifier.py
  
  # Aufbau der Datenbasis
  
  Unsere Datenbasis wurde aufgeteilt in Trainings- und Testdaten. Hierfür wurde eine Ordnerstruktur aufgebaut, die Test- und Trainingsdaten jeweils in
  Datensätze zu den einzelnen von uns behandelten Epochen aufteilt. Während des testens und trainierens wird über diese Ordnerstruktur iteriert. 
  
  Unsere Daten Finden Sie im oben verlinkten Google Drive.
  
  # Viel Spass mit Shazart!
  
