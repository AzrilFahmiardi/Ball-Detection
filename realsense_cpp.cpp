#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Parameter deteksi warna oranye untuk bola RoboCup (rentang diperlebar)
Scalar lower_orange(0, 120, 100);
Scalar upper_orange(20, 255, 255);

// Elemen struktural untuk operasi morfologi
Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

// Informasi intrinsik kamera (diperoleh dari proses kalibrasi)
Mat camera_matrix = (Mat_<double>(3, 3) << 607.1673, 0, 319.5, 0, 607.1673, 239.5, 0, 0, 1);
Mat distortion_coeffs = (Mat_<double>(1, 5) << -0.0461, 0.1772, 0, 0, -0.2892);

// Fungsi untuk menghitung koordinat 3D bola dalam sistem koordinat kamera
Vec3f calculate_3d_position(const Mat& camera_matrix, const Point& center, float depth) {
    double fx = camera_matrix.at<double>(0, 0);
    double fy = camera_matrix.at<double>(1, 1);
    double cx = camera_matrix.at<double>(0, 2);
    double cy = camera_matrix.at<double>(1, 2);

    double z = depth / 1000.0;  // Konversi ke meter
    double x_camera = (cx - center.x) * z / fx;
    double y_camera = (cy - center.y) * z / fy;

    return Vec3f(z, x_camera, y_camera);
}

// Fungsi untuk menghitung jarak horizontal bola dari kamera
float calculate_ground_distance(float z) {
    return z;
}

int main() {
    // Konfigurasi pipeline RealSense
    rs2::pipeline pipeline;
    rs2::config config;
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    pipeline.start(config);

    // Align depth frame ke color frame
    rs2::align align(RS2_STREAM_COLOR);

    while (true) {
        // Baca frame dari kamera RealSense
        rs2::frameset frames = pipeline.wait_for_frames();
        rs2::frameset aligned_frames = align.process(frames);
        rs2::video_frame color_frame = aligned_frames.get_color_frame();
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();
        if (!color_frame || !depth_frame)
            continue;

        // Konversi frame ke OpenCV Mat
        Mat color_image(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        Mat depth_image(Size(640, 480), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);

        // Preprocessing
        GaussianBlur(color_image, color_image, Size(5, 5), 0);

        // Konversi ke HSV
        Mat hsv;
        cvtColor(color_image, hsv, COLOR_BGR2HSV);

        // Masking untuk warna oranye
        Mat mask;
        inRange(hsv, lower_orange, upper_orange, mask);
        erode(mask, mask, kernel);
        dilate(mask, mask, kernel);

        // Mencari contours
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Inisialisasi variabel untuk menyimpan informasi bola
        bool ball_detected = false;
        Vec3f ball_3d;
        float ground_distance = 0.0;

        if (!contours.empty()) {
            // Mengambil kontur terbesar
            auto c = max_element(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) < contourArea(b);
            });

            double area = contourArea(*c);
            if (area > 100) {
                Point2f center;
                float radius;
                minEnclosingCircle(*c, center, radius);

                // Ambil jarak dari depth image
                float distance = depth_frame.get_distance(center.x, center.y);

                if (distance > 0) {
                    ball_3d = calculate_3d_position(camera_matrix, center, distance * 1000);
                    ground_distance = calculate_ground_distance(ball_3d[0]);
                    ball_detected = true;
                }
            }
        }

        Mat result = color_image.clone();
        if (ball_detected) {
            circle(result, Point(ball_3d[1], ball_3d[2]), (int)ball_3d[0], Scalar(0, 255, 0), 2);
            putText(result, "x: " + to_string(ball_3d[0]) + " m", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            putText(result, "y: " + to_string(ball_3d[1]) + " m", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            cout << "x: " << ball_3d[0] << " m, y: " << ball_3d[1] << " m" << endl;
        }

        imshow("Result with Circle, Distance, and Position", result);
        imshow("Mask", mask);
        imshow("Original", color_image);

        if (waitKey(1) == 'q')
            break;
    }

    pipeline.stop();
    destroyAllWindows();
    return 0;
}
