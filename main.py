import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D

# Metody konfiguracyjne środowiska TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

"""
  Zbiór parametrów urzytych w projekcie

  :param int iloscKlas: Ilość wykrytych klas w struktorze zbioru danych
  :param str stosunekZdjec: Wartość stosunku zdjęć walidujących i testujacych do zdjęć uczących
  :param str sciezkaZdjec: Nazwa katalogu z zdjęciami
  :param str nazwaModelu: Nazwa pod jaka zapisywany będzie wytrenowany model sieci
  :param int bokZdjecia: Wartość wysokości i szerokści zdjęć podawana w pikselach
  :param tuple wymiarZdjęc: Wymiar zdjęć wejściowych, szerokośc, wysokość i kolory
  :param int iloscEpok: Ilość epok podczas których uczony jest model
  :param int krokiNaEpoke: Ilość pojedynczych akcji podejmowanych podczas jednej epoki uczenia
  :param int wielkoscPartii: Ilość jednocześnie podawanych elkemetów podczas uczenia
  :param int iloscFiltrow: Ilość filtrów warstw splotowej
  :param tuple wymiarFiltru: Wymiar filtrów warstw splotowej
  :param tuple wymiarFiltru2: Wymiar filtrów warstw splotowej
  :param tuple wymiarPooling: Wymiar macierzy wykorzystywanej w warstwie Pooling redukującej wymiar zdjęć 
  :param int liczbaNauronow: Liczba neuronów warstw ukrytych
"""

# Import
iloscKlas = 0  # to dla skalowalnosci aplikacji
stosunekZdjec = 0.2
sciezkaZdjec = "zdjecia"
nazwaModelu = "wytrenowany.h5"

# Wstępne przetwarzanie zdjęć
bokZdjecia = 150
wymiarZdjec = (bokZdjecia, bokZdjecia, 3)

# Parametry wartsw sieci
iloscFiltrow = 100
wymiarFiltru = (5, 5)
wymiarFiltru2 = (3, 3)
wymiarPooling = (2, 2)
liczbaNauronow = 500

# Uczenie
iloscEpok = 50
krokiNaEpoke = 20
wielkoscPartii = 15


def stworzZbiory():
    """
    Metoda odpowiedzilana za import danych z odpowiednio stoworzonej struktury katalogów,
    Po zaimportowaniu metoda dzieli znalezione zdjęcia na zbiór uczący, testujący i walidujący
    wdług proporcji zadlekarowanej w zmiennej "stosunekZdjec"

    :return: Zbiór uczący, testujący i walidujacy zdjęć wraz z etykietami
    """

    global iloscKlas
    zdjecia = []
    licznikZdjec = 0
    identyfikatoryKlas = []
    listaKatalogowZdjec = os.listdir(path=sciezkaZdjec)
    iloscKlas = len(listaKatalogowZdjec)

    for i in range(0, iloscKlas):
        sciezkaObecnegoKatalogu = os.listdir(sciezkaZdjec + "/" + str(licznikZdjec))
        for j in sciezkaObecnegoKatalogu:
            obecneZdjecie = cv2.imread(sciezkaZdjec + "/" + str(licznikZdjec) + "/" + j)
            obecneZdjecie = cv2.resize(obecneZdjecie, (wymiarZdjec[0], wymiarZdjec[1]))
            zdjecia.append(obecneZdjecie)
            identyfikatoryKlas.append(licznikZdjec)
        licznikZdjec += 1
    npZdjecia = np.array(zdjecia)
    npKlasy = np.array(identyfikatoryKlas)

    X_train, X_test, y_train, y_test = train_test_split(npZdjecia, npKlasy, test_size=stosunekZdjec)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=stosunekZdjec)

    return X_train, y_train, X_test, y_test, X_validation, y_validation


def wstepnePrzetwarzanie(zdjecie):
    """
    Metoda przetwarzająca zbiory zdjęć w celu poprawienia nauki sieci neuronowej.
    W skłąd akcji jakim poddawane jest zdjęcie wchodzi:
        -Zmeina kolorystyki na odcienie szarości co zmeinia ostatni wymiar zdjęcia z wartości 3 na 1
        -Poprawa kontrastu
        -Zmiana zakresu wartości koloru do wartości z przedziału 0-1

    :param np.array zdjecie: Zbiór zdjęć do przerobienia
    :return: Zdjęcie podddane procesom przetwarzania
    """

    zdjecie = cv2.cvtColor(zdjecie, cv2.COLOR_BGR2GRAY)
    zdjecie = cv2.equalizeHist(zdjecie)
    zdjecie = zdjecie / 255

    return zdjecie


def stworzModel():
    """
    Metoda generujaca model seici neuronowej

    :return: Skompoilowany model sieci neuronowej
    """

    model = Sequential()
    model.add((Conv2D(iloscFiltrow, wymiarFiltru, input_shape=(wymiarZdjec[0], wymiarZdjec[1], 1), activation='relu')))
    model.add((Conv2D(iloscFiltrow, wymiarFiltru2, activation='relu')))
    model.add(MaxPooling2D(pool_size=wymiarPooling))

    model.add((Conv2D(iloscFiltrow // 2, wymiarFiltru, activation='relu')))
    model.add((Conv2D(iloscFiltrow // 2, wymiarFiltru2, activation='relu')))
    model.add(MaxPooling2D(pool_size=wymiarPooling))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(liczbaNauronow, activation='relu'))
    model.add(Dense(liczbaNauronow, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(iloscKlas, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def stworzModelTestowy():
    model = Sequential()
    model.add(Flatten(input_shape=(wymiarZdjec[0], wymiarZdjec[1], 1)))

    model.add(Dense(liczbaNauronow, activation='relu'))
    model.add(Dense(liczbaNauronow, activation='relu'))
    model.add(Dense(liczbaNauronow, activation='relu'))

    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(iloscKlas, activation='softmax'))  # OUTPUT LAYER
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, X_validation, y_validation = stworzZbiory()

    X_train = np.array(list(map(wstepnePrzetwarzanie, X_train)))
    X_validation = np.array(list(map(wstepnePrzetwarzanie, X_validation)))
    X_test = np.array(list(map(wstepnePrzetwarzanie, X_test)))

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    IDG = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
    IDG.fit(X_train)
    batches = IDG.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    y_train = to_categorical(y_train, iloscKlas)
    y_validation = to_categorical(y_validation, iloscKlas)
    y_test = to_categorical(y_test, iloscKlas)

    # Turaj wybiaramy model sieci neuronowej
    model = stworzModel()
    print(model.summary())

    callback = ModelCheckpoint(nazwaModelu,
                               monitor='accuracy',
                               save_best_only=True,
                               mode='max')

    history = model.fit(IDG.flow(X_train, y_train, batch_size=wielkoscPartii),
                        steps_per_epoch=krokiNaEpoke,
                        epochs=iloscEpok,
                        validation_data=(X_validation, y_validation),
                        shuffle=1,
                        callbacks=[callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Acurracy')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])
    cv2.waitKey(0)
