#include <rclcpp/rclcpp.hpp>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

class CameraNode : public rclcpp::Node
{
public:
    CameraNode() : Node("camera_node")
    {
        auto qos = rclcpp::QoS(1).best_effort();

        compressed_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/camera/rgb/compressed", qos);

        raw_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/camera/rgb/raw", 10);

        // Reset any stale device connections
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            RCLCPP_ERROR(this->get_logger(), "No RealSense device found!");
            throw std::runtime_error("No RealSense device");
        }

        // Hardware reset to clear any stale locks from previous runs
        devices[0].hardware_reset();
        RCLCPP_INFO(this->get_logger(), "Camera reset, waiting for reinitialize...");
        std::this_thread::sleep_for(std::chrono::seconds(3)); // wait for reset

        // Configure RealSense pipeline
        cfg_.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

        pipe_.start(cfg_);

        compressed_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // ~30 Hz
            std::bind(&CameraNode::loop, this));
    }

private:
    rs2::pipeline pipe_;
    rs2::config cfg_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr raw_pub_;

    rclcpp::TimerBase::SharedPtr compressed_timer_;

    void loop()
    {
        rs2::frameset frames = pipe_.wait_for_frames();
        rs2::frame color = frames.get_color_frame();

        if (!color)
            return;

        cv::Mat frame(
            cv::Size(640, 480),
            CV_8UC3,
            (void*)color.get_data(),
            cv::Mat::AUTO_STEP
        );

        // --- Publish compressed ---
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 60});

        sensor_msgs::msg::CompressedImage comp_msg;
        comp_msg.header.stamp = this->now();
        comp_msg.format = "jpeg";
        comp_msg.data = buffer;

        compressed_pub_->publish(comp_msg);

        // --- Publish raw ---
        auto raw_msg = cv_bridge::CvImage(
            std_msgs::msg::Header(),
            "bgr8",
            frame
        ).toImageMsg();

        raw_msg->header.stamp = this->now();

        raw_pub_->publish(*raw_msg);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
