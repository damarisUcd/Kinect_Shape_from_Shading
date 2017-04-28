#pragma once

struct SolverParameters {
    unsigned int numIter = 1;
    unsigned int nonLinearIter = 3;
    unsigned int linearIter = 200;
    unsigned int patchIter = 32;
    bool profileSolve = true;
    bool optDoublePrecision = false;
};
