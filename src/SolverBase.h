#pragma once
#include "NamedParameters.h"
class SolverBase {
public:
    SolverBase() {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters) {
        fprintf(stderr, "No solve implemented\n");
        return m_finalCost;
    }
    double finalCost() const {
        return m_finalCost;
    }
protected:
    double m_finalCost = nan("");
};
