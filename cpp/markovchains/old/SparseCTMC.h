//
// Created by Michele Fornino on 5/5/18.
// This is the header file of the class : DenseCTMC, which provides methods for Continuous Time Markov Chains. In
// particular, it solves the Kolmogorov Forward Equations to find the stationary distribution of the dynamical system,
// and it provides an easy to use tool to simulate sample paths given an arbitrary number of initial conditions.
//
// IMPORTANT: NEED ROWMAJOR AS STORAGE FOR SPARSE MATRIX Q!!!

#ifndef MARKOVCHAINS_SPARSECTMC_H
#define MARKOVCHAINS_SPARSECTMC_H

#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <eigen/3.3.4/include/eigen3/Eigen/Sparse>
#include <eigen/3.3.4/include/eigen3/Eigen/SparseLU>
#include <vector>
#include "SimCTMC.h"

using namespace Eigen;
using namespace std;

typedef Triplet<double> T;
typedef SparseMatrix<double, ColMajor> SM;
typedef SparseMatrix<double, RowMajor> SMr;

class SparseCTMC
{
private:
    SMr m_Q;
    VectorXd m_S;
    long m_number_of_states;

public:

    // Define a convenience structure to hold the output of the Montecarlo Simulations
    // TODO!!! Make it a class of its own
//    struct SimCTMC
//    {
//        VectorXd m_time_vec;
//        VectorXd m_state_vec;
//        VectorXi m_state_vec_idx;
//        long m_number_of_draws_per_simulation;
//
//        // Explicitly add default constructor
//        SimCTMC() = default;
//
//        // Constructor
//        SimCTMC(VectorXd time_vec, VectorXd state_vec, VectorXi state_vec_idx)
//        {
//            m_number_of_draws_per_simulation = time_vec.rows();
//            m_time_vec = time_vec;
//            m_state_vec = state_vec;
//            m_state_vec_idx = state_vec_idx;
//        };
//
//        // Add methods to save to file
//        // TODO
//
//        // Add operator overloading for << and >> (cin cout)
//        // TODO
//    };

    // Type definition to collate SimCTMC objects. Useful because it is the output of class methods
    typedef vector<SimCTMC> ManySimCTMC;

    // Constructors Definitions
    SparseCTMC() = default;
    explicit SparseCTMC(SMr Q);
    SparseCTMC(SMr Q, VectorXd S);
    void setSparseCTMC(SMr Q, VectorXd S);

    // Destructor Definitions
    ~SparseCTMC() = default;

    // Methods to access data members (trivial functions only)
    SMr getInfinitesimalGenerator(){ return m_Q; }
    VectorXd getStateSpace(){ return m_S; }

    // Methods Definitions
    VectorXd getStationaryDistribution(double step_size = 1000.0);
    ManySimCTMC simSparseCTMC(VectorXi starting_points = VectorXi::LinSpaced(1,0,0), int length_of_simulations = 1000);
};


#endif
