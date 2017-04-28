#include "main.h"
#include "../../shared/OptSolver.h"
#include "SFSSolverInput.h"

int main(int argc, const char * argv[])
{
    std::string inputFilenamePrefix = "../data/shape_from_shading/default";

    SFSSolverInput solverInputGPU;
    solverInputGPU.load(inputFilenamePrefix, true); //On GPU
    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");

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

    std::vector<SolverIteration> solverIterations;
    optSolver->solve(solverParams, problemParams, false, solverIterations);

    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    printf("Save\n");

	return 0;
}
