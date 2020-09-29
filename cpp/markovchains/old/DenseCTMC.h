//
// Created by Michele Fornino on 5/5/18.
// This is the header file of the class : DenseCTMC, which provides methods for Continuous Time Markov Chains. In
// particular, it solves the Kolmogorov Forward Equations to find the stationary distribution of the dynamical system,
// and it provides an easy to use tool to simulate sample paths given an arbitrary number of initial conditions.
//

#ifndef MARKOVCHAINS_DENSECTMC_H
#define MARKOVCHAINS_DENSECTMC_H

#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <vector>
#include "SimCTMC.h"

using namespace Eigen;
using namespace std;

class DenseCTMC
{
public:

    // Define a convenience structure to hold the output of the Montecarlo Simulations
    struct SimCTMC
    {
        VectorXd m_time_vec;
        VectorXd m_state_vec;
        VectorXi m_state_vec_idx;
        int m_number_of_draws_per_simulation;

        SimCTMC(VectorXd time_vec, VectorXd state_vec, VectorXi state_vec_idx)
        {
            m_number_of_draws_per_simulation = time_vec.rows();
            m_time_vec = time_vec;
            m_state_vec = state_vec;
            m_state_vec_idx = state_vec_idx;
        };
    };
    typedef vector<SimCTMC> ManySimCTMC;

private:
    MatrixXd m_Q;
    VectorXd m_S;
    long m_number_of_states;


public:

    // Constructors Definitions
    DenseCTMC(MatrixXd Q);
    DenseCTMC(MatrixXd Q, VectorXd S);
    void setDenseCTMC(MatrixXd Q, VectorXd S);

    // Destructor Definitions
    ~DenseCTMC(){ }

    // Methods to access data members (trivial functions only)
    MatrixXd getInfinitesimalGenerator(){ return m_Q; }
    VectorXd getStateSpace(){ return m_S; }

    // Methods Definitions
    VectorXd getStationaryDistribution(double step_size = 1000.0);
    ManySimCTMC simDenseCTMC(VectorXi starting_points = VectorXi::LinSpaced(1,0,0), int length_of_simulations = 1000);
};


#endif
