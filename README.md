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
  - Einen Link zu unserem gesamten Datensatz auf Google Drive (Link: https://drive.google.com/drive/folders/1nhMymWBEi--4E7iUL7zG4FPGSgMOlZe9?usp=sharing)

Um das Modell auszuführen BITTE BEACHTEN: Die Daten sind aufgrund von Uploadbeschränkungen seitens GitHub nicht im Repo enthalten. Unter diesem https://drive.google.com/drive/folders/1nhMymWBEi--4E7iUL7zG4FPGSgMOlZe9?usp=sharing Link lässt sich der Ordner multi_epochen_data herunterladen. Diesen im Ordner "Shazart_ohneDatensätze" bitte in der 2. Ebene einbinden. Diese ist markiert durch ein leeres File Namens "PLACEHOLDER". Anschließend lässt sich das CNN ausführen.
Wenn Sie die WebApp verwenden wollen (eine Weboberfläche die es dem User erlaubt über eine komfortable Oberfläche Bilder an das CNN zu geben, welches dann seine Predictions ausgibt) bitte den Ordner shazart_web_app in einem beliebigen python editor öffnen und anschließend die Datei run.py starten. Anschließend der im Terminal angegeben IP zur Weboberfläche folgen. Die Web App arbeitet mit unserem zuletzt trainierten Modell.

Wenn Sie lieber das Jupyter Notebook verwenden wollen, bitte den Ordner multi_epochen_data auf der gleichen Ebene wie das Notebook ablegen.

Im Datensatz multi_epochen_data befindet sich neben test und train auch der Ordner "manual_testing". Hier sind ein paar Bilder aus allen Epochen abgelegt, die das CNN noch nicht kennt und mit denen man manuell testen kann, zum Beispiel indem man Sie dem Algorithmus mit der WebApp zur Verfügung stellt.

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
  
