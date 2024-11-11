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

# Parameter deteksi warna oranye untuk bola RoboCup (rentang diperlebar)
lower_orange = np.array([0, 120, 100])   # Lebih rendah agar lebih toleran terhadap variasi pencahayaan
upper_orange = np.array([20, 255, 255])  # Lebih tinggi untuk toleransi

# Elemen struktural untuk operasi morfologi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Konstanta kamera
CAMERA_HEIGHT = 0.4  # Tinggi kamera dari lantai (m)
CAMERA_PITCH = np.deg2rad(30)  # Sudut pitch kamera (derajat)

# Informasi intrinsik kamera (diperoleh dari proses kalibrasi)
camera_matrix = np.array([[607.1673, 0, 319.5],
                          [0, 607.1673, 239.5],
                          [0, 0, 1]])
distortion_coeffs = np.array([-0.0461, 0.1772, 0, 0, -0.2892])

def calculate_3d_position(camera_matrix, center, depth):
    """
    Menghitung koordinat 3D bola dalam sistem koordinat kamera.
    """
    x, y = center
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    
    z = depth / 1000.0  # Konversi ke meter
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    return np.array([x, y, z])

def calculate_ground_distance(z, pitch):
    """
    Menghitung jarak horizontal bola dari kamera dengan mempertimbangkan sudut pitch.
    """
    return z * np.cos(pitch)

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
    color_image = cv2.GaussianBlur(color_image, (5, 5), 0)

    # Konversi ke HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Masking untuk warna oranye
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Mencari contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inisialisasi variabel untuk menyimpan informasi bola
    ball_detected = False
    ball_3d = None
    ground_distance = None

    # Jika ada kontur yang terdeteksi
    if contours:
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
            
            # Hanya lanjutkan jika distance valid
            if distance > 0:
                # Hitung koordinat 3D bola dalam sistem koordinat kamera
                ball_3d = calculate_3d_position(camera_matrix, center, distance)
                z_distance = ball_3d[2]
                
                # Hitung ground distance menggunakan z_distance dan CAMERA_PITCH
                ground_distance = calculate_ground_distance(z_distance, CAMERA_PITCH)
                
                # Cek apakah bola benar-benar berada di dalam frame
                if (0 <= x < color_image.shape[1] - 10) and (0 <= y < color_image.shape[0]):
                    ball_detected = True

    # Gambar lingkaran dan tampilkan posisi bola jika terdeteksi
    result = color_image.copy()
    if ball_detected:
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        # Tampilkan posisi x, y, z, dan jarak bola
        cv2.putText(result, f"x: {ball_3d[0]:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result, f"y: {ball_3d[1]:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result, f"z: {ball_3d[2]:.2f} m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result, f"Ground Distance: {ground_distance:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Print untuk debugging
        print(f"x: {ball_3d[0]:.2f} m, y: {ball_3d[1]:.2f} m, z: {ball_3d[2]:.2f} m, Distance: {ground_distance:.2f} m")
    cv2.imshow('Result with Circle, Distance, and Position', result)
    cv2.imshow('Mask', mask)
    cv2.imshow('Original', color_image)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan resource kamera dan menutup jendela
pipeline.stop()
cv2.destroyAllWindows()