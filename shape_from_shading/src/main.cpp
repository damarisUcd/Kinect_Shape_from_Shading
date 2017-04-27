#include "main.h"
#include "CombinedSolver.h"
#include "SFSSolverInput.h"
#include <signal.h>


#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// Global variables
bool protonect_shutdown =
    false;  ///< Whether the running application should shut down.
bool protonect_paused = false;
libfreenect2::Freenect2Device *devtopause;

void sigint_handler(int s) { protonect_shutdown = true; }

// Doing non-trivial things in signal handler is bad. If you want to pause,
// do it in another thread.
// Though libusb operations are generally thread safe, I cannot guarantee
// everything above is thread safe when calling start()/stop() while
// waitForNewFrame().
void sigusr1_handler(int s) {
  if (devtopause == 0) return;
  /// [pause]
  if (protonect_paused)
    devtopause->start();
  else
    devtopause->stop();
  protonect_paused = !protonect_paused;
  /// [pause]
}

int main(int argc, const char * argv[])
{

// >> Kinect input >> //
  // Camera intrinsics focal length and principal point
  double fx = 0.0, fy = 0.0;
  double cx = 0.0, cy = 0.0;
  bool enable_rgb = true; // /!\ Don't disable both streams !
  bool enable_depth = true;
  string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("Protonect");

  string binpath = "/";

  if (executable_name_idx != string::npos) {
    binpath = program_path.substr(0, executable_name_idx);
  }

  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;

  string serial = "";
  size_t framemax = -1;


  if (freenect2.enumerateDevices() == 0) {
    cout << "no device connected!" << endl;
    return -1;
  }
  serial = freenect2.getDefaultDeviceSerialNumber();

  dev = freenect2.openDevice(serial);

  if (dev == 0) {
    cout << "failure opening device!" << endl;
    return -1;
  }

  devtopause = dev;

  signal(SIGINT, sigint_handler);
#ifdef SIGUSR1
  signal(SIGUSR1, sigusr1_handler);
#endif
  protonect_shutdown = false;

  int types = 0;
  if (enable_rgb)
  {
    types |= libfreenect2::Frame::Color;
  }
  if (enable_depth)
  {
    types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
  }

  libfreenect2::SyncMultiFrameListener listener(types);
  libfreenect2::FrameMap frames;

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (enable_rgb && enable_depth) {
    if (!dev->start()) return -1;
  } else {
    if (!dev->startStreams(enable_rgb, enable_depth)) return -1;
  }

  cout << "device serial: " << dev->getSerialNumber() << endl;
  cout << "device firmware: " << dev->getFirmwareVersion() << endl;

  libfreenect2::Freenect2Device::ColorCameraParams colorParam =
      dev->getColorCameraParams();

  fx = colorParam.fx;
  fy = colorParam.fy;
  cx = colorParam.cx;
  cy = colorParam.cy;

  libfreenect2::Freenect2Device::IrCameraParams depthParam =
      dev->getIrCameraParams();

  libfreenect2::Registration *registration =
      new libfreenect2::Registration(depthParam, colorParam);
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

  size_t framecount = 0;

  while (!protonect_shutdown &&
         (framemax == (size_t)-1 || framecount < framemax)) {
  /*       int i = 0;
   while ( i == 0){
     i++;*/
    if (!listener.waitForNewFrame(frames, 10 * 1000))  // 10 sconds
    {
      cout << "timeout!" << endl;
      return -1;
    }
    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    if (enable_rgb && enable_depth) {
      registration->apply(rgb, depth, &undistorted, &registered);
    }
    cv::Mat depthMat_frame(depth->height, depth->width, CV_32FC1, depth->data);
    cv::Mat depthMat(depth->height, depth->width, CV_32FC1);
    depthMat_frame.copyTo(depthMat);
    cv::Mat rgbMatrix(registered.height, registered.width, CV_8UC4, registered.data);
    cv::Mat initialUnknown;
    depthMat.copyTo(initialUnknown);
  //  resize(rgbMatrix, rgbMatrix, depthMat.size(), 0, 0, INTER_LINEAR);
    cv::cvtColor(rgbMatrix, rgbMatrix, CV_BGRA2GRAY); // transform to gray scale
    cv::Mat mask(2*depth->height, depth->width, CV_32FC1, cv::Scalar(1));

    rgbMatrix.convertTo(rgbMatrix, CV_32FC1);
    double min, max;
    minMaxLoc(depthMat, &min, &max);
    depthMat.convertTo(depthMat, CV_32FC1, -255.0 / max, 255.0);  // Conversion to char to show
    initialUnknown.convertTo(initialUnknown, CV_32FC1, -255.0 / max, 255.0);  // Conversion to char to show

    normalize(rgbMatrix,rgbMatrix,0,1, NORM_L2);
    //normalize(depthMat,depthMat,0,1, NORM_L2);
  //  normalize(initialUnknown,initialUnknown,0,1, NORM_L2);
    cv::Mat depthToShow;
    depthMat.convertTo(depthToShow,CV_8UC1);
    imshow("depthMat", depthToShow);
    cv::waitKey(30);

  //  imshow("RGB",rgbMatrix);
  //  cv::waitKey(30);

  //  std::cout << "colour type " << rgbMatrix.type()<< std::endl;
  //  std::cout << "depth type " << depthMat.type()<< std::endl;

    // >> Kinect input >> //

    // >> OPT >> //
    std::string inputFilenamePrefix = "../data/shape_from_shading/default";
    if (argc >= 2) {
        inputFilenamePrefix = std::string(argv[1]);
    }

    bool performanceRun = false;
    if (argc > 2) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
        }
        else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }

    SFSSolverInput solverInputCPU, solverInputGPU;
    solverInputGPU.load(rgbMatrix,depthToShow, initialUnknown,mask, true);

    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");
    //solverInputCPU.load(inputFilenamePrefix, false);

    CombinedSolverParameters params;
    params.nonLinearIter = 60;
    params.linearIter = 10;
    params.useOpt = true;
    if (performanceRun) {
        params.useCUDA  = false;
        params.useOpt   = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.nonLinearIter = 60;
        params.linearIter = 10;
    }

    CombinedSolver solver(solverInputGPU, params);
    printf("Solving\n");
    solver.solveAll();
    std::shared_ptr<SimpleBuffer> result = solver.result();
    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    printf("Save\n");
    listener.release(frames);
    /**
     * libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
     */
  }

  dev->stop();
  dev->close();

	return 0;

}
