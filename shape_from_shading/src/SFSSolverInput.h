#ifndef SFSSolverInput_h
#define SFSSolverInput_h

#include "SimpleBuffer.h"
#include "TerraSolverParameters.h"
#include "../../shared/NamedParameters.h"
#include <memory>
#include <string>



static  std::shared_ptr<OptImage> createWrapperOptImage(std::shared_ptr<SimpleBuffer> simpleBuffer) {
    std::vector<unsigned int> dims = { (unsigned int)simpleBuffer->width(), (unsigned int)simpleBuffer->height() };
    OptImage::Type t = (simpleBuffer->type() == SimpleBuffer::DataType::FLOAT) ? OptImage::Type::FLOAT : OptImage::Type::UCHAR;
    bool isUnknown = (t == OptImage::Type::FLOAT);
    return std::shared_ptr<OptImage>(new OptImage(dims, simpleBuffer->data(), t, 1, OptImage::Location::GPU, isUnknown, false));
}

struct SFSSolverInput {
    std::shared_ptr<SimpleBuffer>   targetIntensity;
    std::shared_ptr<SimpleBuffer>   targetDepth;
    std::shared_ptr<SimpleBuffer>   initialUnknown; // The values to initialize d_x to before the solver
    std::shared_ptr<SimpleBuffer>   maskEdgeMap; //uint8s, and actually the row and column maps stuck together...
    TerraSolverParameters           parameters;

    void setParameters(NamedParameters& probParams, std::shared_ptr<SimpleBuffer> unknownImage) const {
        probParams.set("w_p", (void*)&parameters.weightFitting);
        probParams.set("w_s", (void*)&parameters.weightRegularizer);
        probParams.set("w_g", (void*)&parameters.weightShading);
        probParams.set("f_x", (void*)&parameters.fx);
        probParams.set("f_y", (void*)&parameters.fy);
        probParams.set("u_x", (void*)&parameters.ux);
        probParams.set("u_y", (void*)&parameters.uy);
        for (int i = 0; i < 9; ++i) {
            char buff[5];
            sprintf(buff, "L_%d", i+1);
            probParams.set(buff, (void*)&(parameters.lightingCoefficients[i]));
        }

        auto unknown = createWrapperOptImage(unknownImage);
        probParams.set("X", unknown);
        probParams.set("D_i", createWrapperOptImage(targetDepth));
        probParams.set("Im", createWrapperOptImage(targetIntensity));
        std::shared_ptr<OptImage> edgeMaskR = createEmptyOptImage(unknown->dims(), OptImage::Type::UCHAR, 1, OptImage::GPU, false);
        std::shared_ptr<OptImage> edgeMaskC = createEmptyOptImage(unknown->dims(), OptImage::Type::UCHAR, 1, OptImage::GPU, false);
        size_t pixCount = initialUnknown->width()*initialUnknown->height();
        edgeMaskR->update(maskEdgeMap->data(), pixCount*sizeof(unsigned char), OptImage::Location::GPU);
        edgeMaskC->update((unsigned char*)maskEdgeMap->data() + pixCount, pixCount*sizeof(unsigned char), OptImage::Location::GPU);
        probParams.set("edgeMaskR", edgeMaskR);
        probParams.set("edgeMaskC", edgeMaskC);
    }

    void load(const cv::Mat frame, const cv::Mat depthMap,const cv::Mat initialUnk,  const cv::Mat mask, bool onGPU) {
      //  targetIntensity = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(frame, onGPU));
      //  maskEdgeMap     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(mask,     onGPU));
      //  initialUnknown  = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(initialUnk, onGPU));
        targetDepth     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(depthMap,     onGPU));
        std::cout << "Data loaded" << std::endl;

        std::cout << targetIntensity->width() << "\n" << targetIntensity->height() << "\n" << targetIntensity->type() << "\n" << std::endl;
        std::cout << initialUnknown->width() << "\n" << initialUnknown->height()<< "\n" << initialUnknown->type() << "\n" << std::endl;

        /*targetIntensity->savePNG("intensity", 255);
        targetDepth->savePNG("depthpng", 1);
        targetDepth->savePLYPoints("depthpoints.ply");
        targetDepth->savePLYMesh("depthmesh.ply");
*/

        //auto test = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetDepth.imagedump", false));
        float* ptr = (float*)targetDepth.get();
        int numActiveUnkowns = 0;
        //targetDepth->savePNG("depth",255);

        /*targetIntensity->savePNG("intensity",255);
        initialUnknown->savePNG("unknown",255);
        maskEdgeMap->savePNG("edge", 1);*/
        /*std::cout << test->type() << std::endl;
        cv::Mat dst(test->width(), test->height(), CV_64FC1,ptr);
        cv::imshow("depth map", dst);
        cv::waitKey(0);*/

        for (int i = 0; i < targetIntensity->width()*targetIntensity->height(); ++i) {
            if (ptr[i] > 0.0f) {
                ++numActiveUnkowns;

            }
        }
        printf("Num Active Unknowns: %d\n", numActiveUnkowns);

        std::string filenamePrefix = "../data/shape_from_shading/default";

        parameters.load(filenamePrefix + ".SFSSolverParameters");
        std::cout << "Parameters loaded" << std::endl;


    }

};

#endif
