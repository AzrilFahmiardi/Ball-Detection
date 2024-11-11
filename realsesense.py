import cv2
import numpy as np
import pyrealsense2 as rs

# Konfigurasi pipeline RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Parameter deteksi warna oranye untuk bola RoboCup
lower_orange = np.array([5, 150, 150])   # H:5, S:150, V:150
upper_orange = np.array([15, 255, 255])  # H:15, S:255, V:255

# Elemen struktural untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    # Baca frame dari kamera RealSense
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    # Konversi frame ke numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Resize, blur, dan konversi ke HSV
    color_image = cv2.resize(color_image, None, fx=0.5, fy=0.5)
    color_image = cv2.GaussianBlur(color_image, (7, 7), 0)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

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

        # Ambil jarak dari depth image
        distance = depth_image[int(y), int(x)]

        # Gambar lingkaran di gambar asli
        result = color_image.copy()
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 2, (0, 0, 255), 3)  # Titik tengah
        cv2.putText(result, f"Distance: {distance} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Tampilkan hasil dengan lingkaran dan jarak
        cv2.imshow('Result with Circle and Distance', result)
    else:
        # Jika tidak ada kontur terdeteksi, tampilkan frame asli
        cv2.imshow('Result with Circle and Distance', color_image)

    # Tampilkan mask dan frame asli untuk debugging
    cv2.imshow('Mask', mask)
    cv2.imshow('Original', color_image)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource kamera dan menutup jendela
pipeline.stop()
cv2.destroyAllWindows()