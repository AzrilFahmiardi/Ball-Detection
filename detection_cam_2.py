import cv2
import numpy as np

# Membuka kamera
cap = cv2.VideoCapture(0)

# Parameter deteksi warna oranye untuk bola RoboCup
lower_orange = np.array([5, 150, 150])   # H:5, S:150, V:150
upper_orange = np.array([15, 255, 255])  # H:15, S:255, V:255

# Elemen struktural untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Parameter kalibrasi untuk estimasi jarak
KNOWN_DISTANCE = 100.0  # Jarak kalibrasi dalam cm
KNOWN_WIDTH = 7.4  # Diameter bola RoboCup dalam cm
KNOWN_PIXEL_WIDTH = None  # Akan diisi saat kalibrasi

def calculate_distance(pixel_width, focal_length):
    """Menghitung jarak berdasarkan lebar pixel yang terdeteksi"""
    distance = (KNOWN_WIDTH * focal_length) / pixel_width
    return distance

def calculate_focal_length(pixel_width, distance):
    """Menghitung focal length kamera"""
    focal_length = (pixel_width * distance) / KNOWN_WIDTH
    return focal_length

# Mode kalibrasi
calibration_mode = True
print("Letakkan bola pada jarak 100cm dan tekan 'c' untuk kalibrasi")

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize, blur, dan konversi ke HSV
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking untuk warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Mencari contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()

    # Jika ada kontur yang terdeteksi
    if len(contours) > 0:
        # Mengambil kontur terbesar
        c = max(contours, key=cv2.contourArea)
        
        # Menghitung area, radius, dan center
        area = cv2.contourArea(c)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Diameter dalam pixel
        pixel_width = 2 * radius

        if calibration_mode:
            # Tampilkan instruksi kalibrasi
            cv2.putText(result, "Tekan 'c' untuk kalibrasi", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif KNOWN_PIXEL_WIDTH is not None:
            # Hitung dan tampilkan jarak
            focal_length = calculate_focal_length(KNOWN_PIXEL_WIDTH, KNOWN_DISTANCE)
            distance = calculate_distance(pixel_width, focal_length)
            
            # Tampilkan informasi
            cv2.putText(result, f"Jarak: {distance:.1f} cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"Diameter: {pixel_width} pixels", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Gambar lingkaran di gambar asli
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 2, (0, 0, 255), 3)  # Titik tengah

    # Tampilkan hasil
    cv2.imshow('Result with Distance', result)
    cv2.imshow('Mask', mask)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and calibration_mode and len(contours) > 0:
        KNOWN_PIXEL_WIDTH = pixel_width
        calibration_mode = False
        print(f"Kalibrasi selesai. Lebar pixel pada jarak {KNOWN_DISTANCE}cm: {KNOWN_PIXEL_WIDTH}")

# Melepaskan resource kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()