#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

int main(int argc, const char* argv[])
{
    std::cout << cv::getBuildInformation() << std::endl;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    return 0;
}
