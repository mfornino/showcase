//
// Created by Michele Fornino on 5/5/18.
// This is the
//

#include "SparseCTMC.h"
#include "SimCTMC.h"
#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <eigen/3.3.4/include/eigen3/Eigen/Sparse>
#include <eigen/3.3.4/include/eigen3/Eigen/SparseLU>
#include <random>

using namespace Eigen;
using namespace std;

// Constructors
SparseCTMC::SparseCTMC(SMr Q)
{
    VectorXd S = VectorXd::LinSpaced(Q.rows(),0,Q.rows()-1);
    setSparseCTMC(Q, S);
}

SparseCTMC::SparseCTMC(SMr Q, VectorXd S)
{
    setSparseCTMC(Q, S);
}

void SparseCTMC::setSparseCTMC(SMr Q, VectorXd S)
{
    // TODO Check Q is a valid infinitesimal generator
    // Must be Square
    //        if (Q.rows() != Q.cols()) {
    //            throw "Infinitesimal Generator must be a Square Matrix.";
    //        }
    // Off diagonal elements must be positive, and rows must sum to 0
    m_Q = Q;

    // State Space
    m_S = S;

    // Grab number of states for convenience
    m_number_of_states = Q.rows();
};

// Nontrivial Methods of the Class
VectorXd SparseCTMC::getStationaryDistribution(double step_size)
{
    // This Method Solves the KF Equations, a.k.a. the Fokker-Planck Equations for the CTMC.

    // Initialize distribution
    VectorXd distribution = VectorXd::Constant(m_number_of_states, 1.0/m_number_of_states);
    SM eye(m_number_of_states, m_number_of_states);
    eye.setIdentity();
    SM A(m_number_of_states,m_number_of_states);
    A = eye - step_size * m_Q.transpose();

    // Define Solver: SparseLU
    SparseLU<SM, COLAMDOrdering<int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);


    return solver.solve(distribution);

}

SparseCTMC::ManySimCTMC SparseCTMC::simSparseCTMC(VectorXi starting_points, int length_of_simulations) {
    // Define Random Engine
    std::default_random_engine generator;

    // Allocate output structure
    int number_of_simulations = starting_points.rows();
    SparseCTMC::ManySimCTMC simulations;
    simulations.reserve(number_of_simulations);

    // Run simulations
    for (int k = 0; k < number_of_simulations; k += 1) {
        VectorXd state_vec(length_of_simulations);
        VectorXi state_vec_idx(length_of_simulations);
        state_vec_idx(0) = starting_points(k);
        state_vec(0) = m_S(state_vec_idx(0));
        VectorXd time_vec(length_of_simulations);
        for (int t = 1; t < length_of_simulations; t += 1) {
            double mindraw = 1e10;
            int minidx = 0;
            double draw;
            for (SMr::InnerIterator it(m_Q,state_vec_idx(t - 1)); it; ++it)
            {
                if (it.col() != it.row()) {
                    exponential_distribution<double> distribution(1.0 / it.value());
                    draw = distribution(generator);
                    if (draw < mindraw) {
                        mindraw = draw;
                        minidx = it.col();
                    }
                }
            }
            state_vec_idx(t) = minidx;
            state_vec(t) = m_S(minidx);
            time_vec(t - 1) = mindraw;
        }
        simulations.emplace_back(SimCTMC(time_vec, state_vec, state_vec_idx));
    }
    return simulations;
}