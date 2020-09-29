//
// Created by Michele Fornino on 5/5/18.
// This is the
//

#include "DenseCTMC.h"
#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

// Constructors
DenseCTMC::DenseCTMC(MatrixXd Q)
{
    VectorXd S = VectorXd::LinSpaced(Q.rows(),0,Q.rows()-1);
    setDenseCTMC(Q, S);
}

DenseCTMC::DenseCTMC(MatrixXd Q, VectorXd S)
{
    setDenseCTMC(Q, S);
}

void DenseCTMC::setDenseCTMC(MatrixXd Q, VectorXd S)
{
    // TODO Check Q is a valid infinitesimal generator
    // Must be Square
    //        if (Q.rows() != Q.cols()) {
    //            throw "Infinitesimal Generator must be a Square Matrix.";
    //        }
    // Off diagonal elements must be positive, and rows must sum to 0
    m_Q = Q;

    // Initialize State Space
    m_S = S;

    // Grab number of states for convenience
    m_number_of_states = Q.rows();
};

// Nontrivial Methods of the Class
VectorXd DenseCTMC::getStationaryDistribution(double step_size)
{
    // This Method Solves the KF Equations, a.k.a. the Fokker-Planck Equations for the CTMC.

    // Initialize distribution
    double uniform = 1.0 / m_number_of_states;
    VectorXd distribution = VectorXd::Constant(m_number_of_states, 1.0/m_number_of_states);
    MatrixXd eye = MatrixXd::Identity(m_number_of_states,m_number_of_states);
    MatrixXd A(m_number_of_states,m_number_of_states);
    A = eye - step_size * m_Q.transpose();

    // Solve System for Stationary Distribution in One Step
    return A.partialPivLu().solve(distribution);
}

DenseCTMC::ManySimCTMC DenseCTMC::simDenseCTMC(VectorXi starting_points, int length_of_simulations)
{
    // Define Random Engine
    std::default_random_engine generator;

    // Allocate output structure
    int number_of_simulations = starting_points.rows();
    DenseCTMC::ManySimCTMC simulations;
    simulations.reserve(number_of_simulations);

    // Run simulations
    for (int k = 0; k < number_of_simulations; k += 1){
        VectorXd state_vec(length_of_simulations);
        VectorXi state_vec_idx(length_of_simulations);
        state_vec_idx(0) = starting_points(k);
        state_vec(0) = m_S(state_vec_idx(0));
        VectorXd time_vec(length_of_simulations);
        for (int t = 1; t < length_of_simulations; t += 1) {
            double mindraw = 1e10;
            int minidx = 0;
            double draw;
            for(int i = 0; i < m_number_of_states; i += 1) {
                if (i != state_vec(t-1)) {
                    exponential_distribution<double> distribution(1.0 / m_Q(state_vec(t-1), i));
                    draw = distribution(generator);
                    if (draw < mindraw) {
                        mindraw = draw;
                        minidx = i;
                    }
                }
            }
            state_vec_idx(t) = minidx;
            state_vec(t) = m_S(minidx);
            time_vec(t-1) = mindraw;
        }
        simulations.emplace_back(SimCTMC(time_vec, state_vec, state_vec_idx));
    }
    return simulations;
}