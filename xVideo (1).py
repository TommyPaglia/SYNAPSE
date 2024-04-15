import cv2
import numpy as np
import time
# Carica il modello di rilevamento del volto
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Avvia lo streaming video
cap = cv2.VideoCapture('http://192.168.1.155:8080/video')

while True:
    # Leggi il frame corrente
    ret, frame = cap.read()
    if not ret:
        break

    # Prepara l'immagine per la classificazione creando un blob 4D
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Passa il blob attraverso la rete e ottieni le rilevazioni dei volti
    net.setInput(blob)
    detections = net.forward()

    # Per ogni volto rilevato, disegna un rettangolo intorno al volto
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, w, h) = box.astype("int")
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2
            print(f"{center_x}, {center_y}")
        

    # Mostra l'immagine
    cv2.imshow("Video", frame)

    # Interrompi il ciclo se 'q' è premuto
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia l'oggetto VideoCapture e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
