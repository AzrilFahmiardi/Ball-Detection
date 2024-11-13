#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <vector>

int main() {
    // Inisialisasi pipeline RealSense
    rs2::pipeline p;
    rs2::config cfg;

    // Enable depth and color stream
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    // Start pipeline
    p.start(cfg);

    // Create alignment object to align depth frame to color frame
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // Warna oranye untuk bola RoboCup
    cv::Scalar lower_orange(0, 120, 100);   // Lebih rendah untuk variasi pencahayaan
    cv::Scalar upper_orange(20, 255, 255);  // Lebih tinggi untuk toleransi

    // Elemen struktural untuk operasi morfologi
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // Kamera matrix dan koefisien distorsi
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 607.1673, 0, 319.5,
                                                      0, 607.1673, 239.5,
                                                      0, 0, 1);
    cv::Mat distortion_coeffs = (cv::Mat_<double>(1, 5) << -0.0461, 0.1772, 0, 0, -0.2892);

    while (true) {
        // Menunggu dan mengambil frame dari pipeline
        rs2::frameset frames = p.wait_for_frames();
        rs2::frameset aligned_frames = align_to_color.process(frames);

        // Mengambil frame warna dan depth
        rs2::frame color_frame = aligned_frames.get_color_frame();
        rs2::frame depth_frame = aligned_frames.get_depth_frame();

        // Konversi frame ke cv::Mat
        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data());
        cv::Mat depth_image(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data());

        // Preprocessing
        cv::GaussianBlur(color_image, color_image, cv::Size(5, 5), 0);

        // Konversi ke HSV
        cv::Mat hsv;
        cv::cvtColor(color_image, hsv, cv::COLOR_BGR2HSV);

        // Masking untuk warna oranye
        cv::Mat mask;
        cv::inRange(hsv, lower_orange, upper_orange, mask);
        cv::erode(mask, mask, kernel);
        cv::dilate(mask, mask, kernel);

        // Mencari kontur
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Variabel untuk menyimpan informasi bola
        bool ball_detected = false;
        cv::Point2f center;
        float radius = 0;
        float z_distance = 0;

        if (!contours.empty()) {
            // Ambil kontur terbesar
            std::vector<cv::Point> max_contour = *std::max_element(contours.begin(), contours.end(),
                                                                   [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                                       return cv::contourArea(a) < cv::contourArea(b);
                                                                   });

            // Hitung area, radius, dan center
            float area = cv::contourArea(max_contour);
            if (area > 100) {  // Filter berdasarkan luas area kontur
                // Hitung lingkaran pembatas
                cv::minEnclosingCircle(max_contour, center, radius);

                // Ambil jarak dari depth image
                int x = static_cast<int>(center.x);
                int y = static_cast<int>(center.y);
                z_distance = depth_image.at<uint16_t>(y, x) / 10.0f;  // Konversi ke cm

                // Hanya lanjutkan jika jarak valid
                if (z_distance > 0) {
                    // Menandakan bola terdeteksi
                    ball_detected = true;
                }
            }
        }

        // Menampilkan hasil
        cv::Mat result = color_image.clone();
        if (ball_detected) {
            cv::circle(result, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2);
            cv::putText(result, "Z: " + std::to_string(z_distance) + " cm", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Result", result);
        cv::imshow("Mask", mask);

        // Keluar jika tekan 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Hentikan pipeline dan tutup jendela
    p.stop();
    cv::destroyAllWindows();

    return 0;
}
