#include "main.h"
#include "OptSolver.h"
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

void sfs_static(){
    std::string inputPrefix = "data/shape_from_shading/default";

    cv::Mat rgb = imread(inputPrefix+"_targetIntensity.png",CV_LOAD_IMAGE_COLOR);
    rgb.convertTo(rgb, CV_32FC1, 1.0/255.0);
    cv::Mat depth = imread(inputPrefix+"_targetDepth.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat initialUnknown = imread(inputPrefix+"_initialUnknown.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat mask = imread(inputPrefix+"_maskEdgeMap.png",CV_LOAD_IMAGE_COLOR);

    printf("hi\n");

    //imshow("rgb",rgb);

    SFSSolverInput solverInputCPU, solverInputGPU;
	solverInputGPU.load(rgb, depth, initialUnknown, mask, true);

	printf("hi2\n");

	std::string prefix = "output/";
	std::string postfix = std::to_string(solverInputGPU.parameters.weightShading)
				  + "_" + std::to_string(solverInputGPU.parameters.weightFitting)
				  + "_" + std::to_string(solverInputGPU.parameters.weightRegularizer)
				  + "_" + std::to_string(solverInputGPU.parameters.weightPrior);

	solverInputGPU.targetDepth->savePLYMesh(prefix+"sfsInitDepth_"+postfix+".ply");
	solverInputGPU.targetDepth->savePNG(prefix+"sfsInitDepth_"+postfix,255);
	solverInputGPU.targetIntensity->savePNG(prefix+"sfsRGBMatrix_"+postfix,255);
	solverInputGPU.maskEdgeMap->savePNG(prefix+"sfsEdges_"+postfix,255);
	solverInputGPU.initialUnknown->savePNG(prefix+"sfsUnknown_"+postfix,255);

	//solverInputCPU.load(inputFilenamePrefix, false);

	printf("Solving\n");
	//
	std::shared_ptr<SimpleBuffer> result;
	std::vector<unsigned int> dims;
	result = std::make_shared<SimpleBuffer>(*solverInputGPU.initialUnknown, true);
	dims = { (unsigned int)result->width(), (unsigned int)result->height() };
	std::shared_ptr<OptSolver> optSolver = std::make_shared<OptSolver>(dims, "shape_from_shading.t", "gaussNewtonGPU", false);
	NamedParameters solverParams;
	NamedParameters problemParams;
	solverInputGPU.setParameters(problemParams, result);
	unsigned int nonLinearIter = 3;
	unsigned int linearIter = 200;
	solverParams.set("nIterations", &nonLinearIter);

	optSolver->solve(solverParams, problemParams);

	printf("Solved\n");
	printf("About to save\n");
	//result->save(prefix+"sfsOutput_"+postfix+".imagedump");
	result->savePNG(prefix+"sfsOutput_"+postfix, 255.0);
	result->savePLYMesh(prefix+"sfsOutput_"+postfix+".ply");
	printf("Save\n");
}


void computeMask(cv::Mat* mask, cv::Mat* depth, cv::Mat* albedo, float depth_threshold=1.0){


	int rows = (*depth).rows;
	int cols = (*depth).cols;

	// ------------ Depth discontinuities ----------------- //

	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;

	cv::Mat depth0Mask;

	cv::threshold(*depth,depth0Mask,0.0,1.0,0);

	depth0Mask.convertTo(depth0Mask,CV_32FC1);

	cv::Mat gradX, gradY;
	Sobel(*depth,gradX,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
	Sobel(*depth,gradY,ddepth,0,1,3,scale,delta,BORDER_DEFAULT);
	gradX.convertTo(gradX,CV_32FC1);
	gradY.convertTo(gradY,CV_32FC1);

	gradX = gradX.mul(depth0Mask);
	gradY = gradY.mul(depth0Mask);

	vconcat(gradX,gradY,*mask);
	pow(*mask,2,*mask);

	double min, max;
	cv::minMaxLoc(*mask, &min, &max);

	*mask = *mask - cv::Scalar(min);
	*mask = (*mask).mul(cv::Scalar(1.0/(max-min)));

	cv::minMaxLoc(*mask, &min, &max);

	(*mask).convertTo(*mask,CV_32FC1);

	cv::threshold(*mask,*mask,0.0,1.0,1);

	//imshow("Depth_edges",*mask);

	// ------------ Albedo discontinuities ----------------- //


	// needed: lighting coefficients, rgb image, H(n(i,j))

/*	cv::Mat I_c;
	(*rgb).copyTo(I_c);
	I_c.reshape(0,1);

	cv::Mat I_a(rows*cols, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat   H(rows*cols, 9, CV_32FC1, cv::Scalar(0));
	cv::Mat   l(9,1,CV_32FC1,cv::Scalar(0));

	cv::divide(I_c,H.mul(l),I_a);*/

	(*albedo).reshape(0,rows);


	cv::Mat mask2;
	Sobel((*albedo),gradX,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
	Sobel((*albedo),gradY,ddepth,0,1,3,scale,delta,BORDER_DEFAULT);
	gradX.convertTo(gradX,CV_32FC1);
	gradY.convertTo(gradY,CV_32FC1);

	//gradX = gradX.mul(depth0Mask);
	//gradY = gradY.mul(depth0Mask);

	vconcat(gradX,gradY,mask2);
	pow(mask2,2,mask2);

	cv::minMaxLoc(mask2, &min, &max);

	mask2 = mask2 - cv::Scalar(min);
	mask2 = mask2.mul(cv::Scalar(1.0/(max-min)));

	cv::minMaxLoc(mask2, &min, &max);

	mask2.convertTo(mask2,CV_32FC1);

	cv::threshold(mask2,mask2,0.0,1.0,1);

	// ----------- mask indicating zero depth (unregistered) --------------
	cv::Mat mask3;

	cv::threshold((*depth),mask3,0.0,1.0,0);
	vconcat(mask3,mask3,mask3);
	mask3.convertTo(mask3,CV_32FC1);


	// ----------- mask indicating foreground using depth_threshold --------------
	cv::Mat mask4;

	cv::threshold((*depth),mask4,depth_threshold,1.0,1);

	//(*depth) = (*depth).mul(mask4);

	vconcat(mask4,mask4,mask4);
	mask4.convertTo(mask4,CV_32FC1);

	// --------- Combine Masks -------------

	(*mask) = mask2.mul(*mask);
	(*mask) = mask3.mul(*mask);

	// mask4 affects result you would not expect
	(*mask) = mask4.mul(*mask);

	//imshow("Albedo edges",*mask);

}

int main(int argc, const char * argv[])
{

    // >> Kinect input >> //
      // Camera intrinsics focal length and principal point
      double fx = 0.0, fy = 0.0;
      double cx = 0.0, cy = 0.0;

      string program_path(argv[0]);
      size_t executable_name_idx = program_path.rfind("Protonect");

      string binpath = "/";

      if (executable_name_idx != string::npos) {
        binpath = program_path.substr(0, executable_name_idx);
      }

      libfreenect2::Freenect2 freenect2;
      libfreenect2::Freenect2Device *dev = 0;

      string serial = "";

      bool saveFiles = false;
      bool enable_rgb = true; // /!\ Don't disable both streams !
      bool enable_depth = true;
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

      libfreenect2::Registration *registration = new libfreenect2::Registration(depthParam, colorParam);
      libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

      size_t framecount = 0;

      while (!protonect_shutdown &&
             (framemax == (size_t)-1 || framecount < framemax)) {
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
        cv::Mat rgbMatrix(registered.height, registered.width, CV_8UC4, registered.data);
        cv::Mat binary_mask(2*depth->height, depth->width, CV_32FC1, cv::Scalar(1));
        cv::Mat mask(depth->height, depth->width, CV_32FC1, cv::Scalar(1));
        mask.setTo(0, depthMat_frame == 0);
        cv::Mat targetROI;
        targetROI = binary_mask(cv::Rect(0, 0, mask.cols, mask.rows));
        mask.copyTo(targetROI);
        targetROI = binary_mask(cv::Rect(0, mask.rows, mask.cols, mask.rows));
        mask.copyTo(targetROI);
        //imshow("mask",binary_mask);
      //

        const int filterWidth = 7;
        const int structEltSize1 = 7;
        const int structEltSize2 = 5;
        cv::Mat element1 = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(structEltSize1, structEltSize1),cv::Point(structEltSize1/2,structEltSize1/2));
        cv::Mat element2 = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(structEltSize2, structEltSize2),cv::Point(structEltSize2/2,structEltSize2/2));

      double min, max;

        cv::Mat depthMat,initialUnknown,depthD,depthBilateralF,depthBilateralD;
        depthMat_frame.copyTo(depthMat);

        //dilate(depthMat,depthMat,element1);
        //dilate(rgbMatrix,rgbMatrix,element1);
        //dilate(mask,mask,element1);
        //dilate(binary_mask,binary_mask,element1);

        GaussianBlur(depthMat, depthBilateralF, cv::Size(filterWidth,filterWidth),3,3);

        depthBilateralF.convertTo(depthBilateralD, CV_64FC1);  // Conversion to double
        minMaxLoc(depthBilateralD,&min,&max);


        // Remove spikes from Gaussian blurring by erosion.
        // We should also dilate the depth map before blurring to fill some gaps.
        mask.convertTo(mask,CV_64FC1);

        erode(mask, mask, element1);
        erode(binary_mask, binary_mask, element1);
        //imshow( "Eroded mask", binary_mask);
        depthBilateralD = depthBilateralD.mul(mask);



        // Conversion to uchar (value between 0 and 255) to show
        depthBilateralD.convertTo(depthBilateralF, CV_32FC1, 255.0 / max);

        cv::Mat subt;
      //  resize(rgbMatrix, rgbMatrix, depthMat.size(), 0, 0, INTER_LINEAR);
        cv::cvtColor(rgbMatrix, rgbMatrix, CV_BGRA2GRAY); // transform to gray scale

        rgbMatrix.convertTo(rgbMatrix, CV_32FC1,1.0/255.0);
        mask.convertTo(mask,CV_32FC1);
        rgbMatrix = rgbMatrix.mul(mask);
        //GaussianBlur(rgbMatrix, rgbMatrix, cv::Size(3,3),3,3);

        minMaxLoc(depthMat, &min, &max);
        depthMat.convertTo(depthMat, CV_32FC1, 1.0 / max);  // Conversion to char to show
        cv::Mat depth2show;
        depthMat.convertTo(depth2show,CV_8UC1);
        depthBilateralF.convertTo(depthBilateralF, CV_32FC1, 1.0/255.0);  // Normalize between 0-1

        cv::threshold(depthBilateralF,depthBilateralF,0.5,1.0,4);
        cv::Mat threshMask(depth->height, depth->width, CV_32FC1, cv::Scalar(1));
        cv::threshold(depthBilateralF,threshMask,0,1.0,0);
        rgbMatrix = rgbMatrix.mul(threshMask);
        mask = mask.mul(threshMask);

        targetROI = binary_mask(cv::Rect(0, 0, mask.cols, mask.rows));
        mask.copyTo(targetROI);
        targetROI = binary_mask(cv::Rect(0, mask.rows, mask.cols, mask.rows));
        mask.copyTo(targetROI);


        //imshow("rgb", rgbMatrix);
        cv::waitKey(30);
        depthBilateralF.copyTo(initialUnknown);
        // >> Kinect input >> //

        // >> OPT >> //
        //This remains to load the parameters
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

        // TODO: give computeMask() albedo image instead of rgb
        computeMask(&binary_mask, &depthMat, &rgbMatrix,0.25);

        SFSSolverInput solverInputCPU, solverInputGPU;
        solverInputGPU.load(rgbMatrix, depthBilateralF, initialUnknown, binary_mask, true);

        imshow("rgb",rgbMatrix);
        imshow("depthBilateralF",depthBilateralF);
        imshow("initialUnknown",initialUnknown);
        imshow("binary_mask",binary_mask);

        std::string prefix = "output/";
        std::string postfix = std::to_string((int)solverInputGPU.parameters.weightShading)
        			  + "_" + std::to_string((int)solverInputGPU.parameters.weightFitting)
        			  + "_" + std::to_string((int)solverInputGPU.parameters.weightRegularizer)
        			  + "_" + std::to_string((int)solverInputGPU.parameters.weightPrior);

        solverInputGPU.targetDepth->savePLYMesh(prefix+"sfsInitDepth_"+postfix+".ply");
        solverInputGPU.targetDepth->savePNG(prefix+"sfsInitDepth_"+postfix,255);
        solverInputGPU.targetIntensity->savePNG(prefix+"sfsRGBMatrix_"+postfix,255);
        solverInputGPU.maskEdgeMap->savePNG(prefix+"sfsEdges_"+postfix,255);
        solverInputGPU.initialUnknown->savePNG(prefix+"sfsUnknown_"+postfix,255);

        //solverInputCPU.load(inputFilenamePrefix, false);

        printf("Solving\n");
        //
        std::shared_ptr<SimpleBuffer> result;
        std::vector<unsigned int> dims;
        result = std::make_shared<SimpleBuffer>(*solverInputGPU.initialUnknown, true);
        dims = { (unsigned int)result->width(), (unsigned int)result->height() };
        std::shared_ptr<OptSolver> optSolver = std::make_shared<OptSolver>(dims, "shape_from_shading.t", "gaussNewtonGPU", false);
        NamedParameters solverParams;
        NamedParameters problemParams;
        solverInputGPU.setParameters(problemParams, result);
        unsigned int nonLinearIter = 3;
        unsigned int linearIter = 200;
        solverParams.set("nIterations", &nonLinearIter);

        optSolver->solve(solverParams, problemParams);

        printf("Solved\n");
        printf("About to save\n");
        //result->save(prefix+"sfsOutput_"+postfix+".imagedump");
        result->savePNG(prefix+"sfsOutput_"+postfix, 255.0);
        result->savePLYMesh(prefix+"sfsOutput_"+postfix+".ply");
        printf("Save\n");

        framecount++;

        listener.release(frames);
        /**
         * libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
         */
      }

      dev->stop();
      dev->close();

	return 0;
}
