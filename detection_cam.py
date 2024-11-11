import cv2
import numpy as np

# Membuka kamera
cap = cv2.VideoCapture(0)

# Parameter deteksi warna oranye untuk bola RoboCup
lower_orange = np.array([5, 150, 150])   # H:5, S:150, V:150
upper_orange = np.array([15, 255, 255])  # H:15, S:255, V:255

# Elemen struktural untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize, blur, dan konversi ke HSV
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor  (frame, cv2.COLOR_BGR2HSV)

    # Masking untuk warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Mencari contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Jika ada kontur yang terdeteksi
    if len(contours) > 0:
        # Mengambil kontur terbesar
        c = max(contours, key=cv2.contourArea)

        # Menghitung area, radius, dan center
        area = cv2.contourArea(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)

        # Gambar lingkaran di gambar asli
        result = frame.copy()
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 2, (0, 0, 255), 3)  # Titik tengah

        # Tampilkan hasil dengan lingkaran
        cv2.imshow('Result with Circle', result)
    else:
        # Jika tidak ada kontur terdeteksi, tampilkan frame asli
        cv2.imshow('Result with Circle', frame)

    # Tampilkan mask dan frame asli untuk debugging
    cv2.imshow('Mask', mask)
    cv2.imshow('Original', frame)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()
