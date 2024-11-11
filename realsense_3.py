import cv2
import numpy as np
import pyrealsense2 as rs

# Konfigurasi pipeline RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Align depth frame ke color frame
align = rs.align(rs.stream.color)

# Parameter deteksi warna oranye untuk bola RoboCup
lower_orange = np.array([5, 150, 150])   # H:5, S:150, V:150
upper_orange = np.array([15, 255, 255])  # H:15, S:255, V:255

# Elemen struktural untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Konstanta kamera
CAMERA_HEIGHT = 1.2  # Tinggi kamera dari lantai (m)
CAMERA_PITCH = np.deg2rad(30)  # Sudut pitch kamera (derajat)

# Informasi intrinsik kamera (diperoleh dari proses kalibrasi)
camera_matrix = np.array([[607.1673, 0, 319.5],
                          [0, 607.1673, 239.5],
                          [0, 0, 1]])
distortion_coeffs = np.array([-0.0461, 0.1772, 0, 0, -0.2892])

def calculate_3d_position(camera_matrix, distortion_coeffs, center, depth):
    """
    Menghitung koordinat 3D bola dalam sistem koordinat kamera.
    """
    x, y = center
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    
    z = depth / 1000.0  # Konversi ke meter
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    return np.array([x, y, z])

def transform_to_top_down_view(ball_3d):
    """
    Transformasi koordinat 3D ke koordinat 2D dengan pandangan dari atas (kiri-kanan, depan-belakang).
    """
    x, y, z = ball_3d
    x_top = x
    y_top = z * np.sin(CAMERA_PITCH) + y * np.cos(CAMERA_PITCH)
    return x_top, y_top

while True:
    # Baca frame dari kamera RealSense
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    # Konversi frame ke numpy array
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Preprocessing
    color_image = cv2.resize(color_image, None, fx=0.75, fy=0.75)  # Resizing dengan faktor 0.75
    color_image = cv2.GaussianBlur(color_image, (5, 5), 0)  # Gaussian Blurring dengan kernel 5x5

    # Konversi ke HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Masking untuk warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Mencari contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi variabel untuk menyimpan informasi bola
    ball_detected = False
    ball_3d = None
    ball_2d = None
    distance = None

    # Jika ada kontur yang terdeteksi
    if len(contours) > 0:
        # Mengambil kontur terbesar
        c = max(contours, key=cv2.contourArea)

        # Menghitung area, radius, dan center
        area = cv2.contourArea(c)
        if area > 100:  # Filter berdasarkan luas area kontur
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)

            # Ambil jarak dari depth image
            distance = depth_image[int(y), int(x)]
            distance = distance / 1000.0  # Konversi ke meter

            # Hitung koordinat 3D bola dalam sistem koordinat kamera
            ball_3d = calculate_3d_position(camera_matrix, distortion_coeffs, center, distance)

            # Transformasi ke koordinat 2D dengan pandangan dari atas (kiri-kanan, depan-belakang)
            ball_2d = transform_to_top_down_view(ball_3d)

            ball_detected = True

    # Gambar lingkaran dan tampilkan posisi bola jika terdeteksi
    result = color_image.copy()
    if ball_detected:
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.putText(result, f"Distance: {distance:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(result, f"Position: ({ball_2d[0]:.2f}, {ball_2d[1]:.2f}) m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('Result with Circle, Distance and Position', result)

    # Tampilkan mask dan frame asli untuk debugging
    cv2.imshow('Mask', mask)
    cv2.imshow('Original', color_image)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource kamera dan menutup jendela
pipeline.stop()
cv2.destroyAllWindows()