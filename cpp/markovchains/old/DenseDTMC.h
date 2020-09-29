//
// Created by Michele Fornino on 5/5/18.
// This is the header file of the class : DenseCTMC, which provides methods for Continuous Time Markov Chains. In
// particular, it solves the Kolmogorov Forward Equations to find the stationary distribution of the dynamical system,
// and it provides an easy to use tool to simulate sample paths given an arbitrary number of initial conditions.
//

#ifndef MARKOVCHAINS_DENSEDTMC_H
#define MARKOVCHAINS_DENSEDTMC_H

#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

class DenseDTMC
{
public:

    // Define a convenience structure to hold the output of the Montecarlo Simulations
    struct SimDTMC
    {
        VectorXi m_time;
        VectorXd m_state_vec;
        VectorXi m_state_vec_idx;
        int m_number_of_draws_per_simulation;

        SimDTMC(VectorXd state_vec, VectorXi state_vec_idx)
        {
            int number_of_draws_per_simulation = state_vec.rows();
            m_number_of_draws_per_simulation = number_of_draws_per_simulation;
            m_time = VectorXi::LinSpaced(number_of_draws_per_simulation,0,number_of_draws_per_simulation-1);
            m_state_vec = state_vec;
            m_state_vec_idx = state_vec_idx;
        };
    };
    typedef vector<SimDTMC> ManySimDTMC;

private:
    MatrixXd m_P;
    VectorXd m_S;
    int m_number_of_states;

public:

    // Constructors Definitions
    DenseDTMC(){};
    DenseDTMC(MatrixXd P);
    DenseDTMC(MatrixXd P, VectorXd S);
    void setDenseDTMC(MatrixXd P, VectorXd S);

    // Static Class Definition Methods (convenience functions)
    static DenseDTMC discretizeAR1_Rouwenhorst(double rho = 0.8, double sigma_eps = 0.1, double mu = 0, int number_of_states = 9);
//    static DenseDTMC discretizeAR1_Tauchen(double rho = 0.9, double sigma_eps = 0.1, double stdev_to_approximate = 3, int number_of_states);

    // Destructor Definition
    ~DenseDTMC(){}

    // Methods to access data members (trivial functions only)
    MatrixXd getTransitionMatrix(){ return m_P; }
    VectorXd getStateSpace(){ return m_S; }

    // Methods Definitions
    VectorXd getStationaryDistribution(bool iterative = true, double criterion_iterative = 1e-10, int maxit = 1000);
    ManySimDTMC simDenseDTMC(VectorXi starting_points = VectorXi::LinSpaced(1,0,0), int length_of_simulations = 1000);
};

#endif
