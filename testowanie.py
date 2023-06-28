import cv2
import main
import numpy as np
import keras.saving.saved_model_experimental

"""
  Zbiór parametrów urzytych w projekcie

  :param float minPrawdo: Zmienna okreslające wartość prawdopodobieństwa przy której zostaje wyświetlona nazwa klasy
  :param enum czcionka: zmienna okreslajaca czcionke tekstu wyświtlanego w oknie programu
  :param int bokZdjecUczacych: Wartość wysokości i szerokści zdjęć podawana w pikselach
"""

#Parametry programu
minPrawdo = 0.94
czcionka = cv2.FONT_HERSHEY_COMPLEX
bokZdjecUczacych = main.bokZdjecia

#Konfiguracja kamery
kam = cv2.VideoCapture(0)
kam.set(3, 1280)
kam.set(4, 720)

#Import wytrenowanego modelu
model = keras.models.load_model(main.nazwaModelu)

while True:

    _, obrazKamery = kam.read()
    przetworzonyObraz = np.asarray(obrazKamery)
    przetworzonyObraz = cv2.resize(przetworzonyObraz, (bokZdjecUczacych, bokZdjecUczacych))
    przetworzonyObraz = main.wstepnePrzetwarzanie(przetworzonyObraz)
    cv2.imshow("Przetworzony obraz", cv2.flip(przetworzonyObraz, 1))
    przetworzonyObraz = przetworzonyObraz.reshape(1, bokZdjecUczacych, bokZdjecUczacych, 1)

    predictions = model.predict(przetworzonyObraz)
    print(predictions)

    probabilityValue = np.amax(predictions)
    obrazKamery = cv2.flip(obrazKamery, 1)
    if predictions[0][0] > predictions[0][1]:
        label = 'Banan'
    else:
        label = 'Jablko'
    if probabilityValue > minPrawdo:
        cv2.putText(obrazKamery, label, (120, 35), czcionka, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(obrazKamery, str(round(probabilityValue * 100, 2)) + "%", (300, 75), czcionka, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(obrazKamery, "Obiekt: ", (20, 35), czcionka, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(obrazKamery, "Prawdopodobienstwo: ", (20, 75), czcionka, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Klasyfikacja w czasie rzeczywistym", obrazKamery)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break