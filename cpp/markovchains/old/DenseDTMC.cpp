//
// Created by Michele Fornino on 5/7/18.
//

#include "DenseDTMC.h"
#include <eigen/3.3.4/include/eigen3/Eigen/Dense>
#include <random>
#include <vector>

using namespace Eigen;
using namespace std;

//// Constructors
DenseDTMC::DenseDTMC(MatrixXd P)
{
    VectorXd S = VectorXd::LinSpaced(P.rows(),0,P.rows()-1);
    setDenseDTMC(P, S);
}

DenseDTMC::DenseDTMC(MatrixXd P, VectorXd S)
{
    setDenseDTMC(P, S);
}

void DenseDTMC::setDenseDTMC(MatrixXd P, VectorXd S)
{
    m_P = P;
    m_S = S;
    m_number_of_states = P.rows();
}


//// Convenience Functions to Initialize the Class without Providing the Transition Matrix
DenseDTMC DenseDTMC::discretizeAR1_Rouwenhorst(double rho, double sigma_eps, double mu, int number_of_states)
{
    // Rouwenhorst Method consists in approximating the AR(1) process:
    // y_t = mu + rho y_t-1 + eps_t
    // by a discrete Markov chain on the set [y_1, y_N], where N is the number of states of the approximating MC and
    // these are chosen to match the variance of the process. It is very well suited to persistent processes, but has
    // the drawback that one is not able to choose the window in which to integrate, as in Tauchen.


    if (number_of_states > 1) {
        // Initialize Heteroskedasticity Guards parameters
        double p = (1 + rho) / 2;
        double q = p;

        // Construct Transition Matrix iteratively.
        MatrixXd tm(2,2);
        tm(0,0) = p; tm(0,1) = 1-p;
        tm(1,0) = 1-q; tm(1,1) = q;
        for (int i = 2; i < number_of_states; i += 1){
            MatrixXd newtm = MatrixXd::Zero(i+1, i+1);
            newtm.topLeftCorner(i,i) += p * tm;
            newtm.topRightCorner(i,i) += (1.0-p) * tm;
            newtm.bottomLeftCorner(i,i) += (1.0-q) * tm;
            newtm.bottomRightCorner(i,i) += q * tm;
            newtm.block(1,0,i-1,i+1) = newtm.block(1,0,i-1,i+1)/2.0;
            tm = newtm;
        }

        // Construct Appropriate State Space
        double psi = sqrt(number_of_states-1) * sigma_eps / sqrt(1 - pow(rho, 2.0));
        VectorXd S = VectorXd::LinSpaced(number_of_states, mu / (1.0-rho) -psi , mu / (1.0-rho) + psi);

        // Set Class Members Values
        DenseDTMC approximating_markov_chain = DenseDTMC(tm, S);
        return approximating_markov_chain;

    } else {

        // Handle case with number of states = 1
        MatrixXd tm(1,1); tm(0,0) = 1;
        DenseDTMC approximating_markov_chain = DenseDTMC(tm);
        return approximating_markov_chain;

    }

}

//void DenseDTMC::discretizeAR1_Tauchen(double rho, double sigma, double stdev_to_approximate, int number_of_states)
//{
//    // TODO
//}


//// Nontrivial Methods of the Class
VectorXd DenseDTMC::getStationaryDistribution(bool iterative, double criterion_iterative, int maxit)
{
    // This Method Solves for the stationary distribution.
    // Initialize
    VectorXd p = VectorXd::Constant(m_number_of_states,1.0/m_number_of_states);

    if (!iterative) {
        //        EigenSolver<MatrixXd> es(m_P, false);
//        VectorXcd lambda = es.eigenvalues();
//
//        //TODO ADD THIS!!!
        VectorXd identity_distribution = VectorXd::Constant(m_number_of_states, 1.0/m_number_of_states);
        return identity_distribution;

    } else if (iterative) {
        MatrixXd PT = m_P.transpose();
        VectorXd p_next;
        double supnorm = 1e10;
        double it = 0;
        while(supnorm > criterion_iterative && it < maxit){
            p_next = PT * p;
            double maxabsdiff = -1.0;
            double absdiff;
            for (int i = 0; i < m_number_of_states; i += 1){
                absdiff = abs(p_next(i) - p(i));
                if (absdiff > maxabsdiff) {
                    maxabsdiff = absdiff;
                }
            }
            p = p_next;
            supnorm = maxabsdiff;
            it += 1;
        }
    }
    return p;
}

DenseDTMC::ManySimDTMC DenseDTMC::simDenseDTMC(VectorXi starting_points, int length_of_simulations)
{
    // Define Random Engine
    std::default_random_engine generator;

    // Allocate output structure
    int number_of_simulations = starting_points.rows();
    DenseDTMC::ManySimDTMC simulations;
    simulations.reserve(number_of_simulations);

    // Run simulations. Use Skorohod theorem.
    // Compute CDF of each conditional transition probability function
    MatrixXd cdfP(m_number_of_states,m_number_of_states);
    for (int j = 0; j < m_number_of_states; j += 1) {
        cdfP.col(j) = m_P.leftCols(j + 1).rowwise().sum();
    }

    // Do simulations
    uniform_real_distribution<double> distribution(0.0,1.0);
    for (int k = 0; k < number_of_simulations; k += 1){
        VectorXd state_vec(length_of_simulations);
        VectorXi state_vec_idx(length_of_simulations);
        state_vec_idx(0) = starting_points(k);
        state_vec(0) = m_S(state_vec_idx(0));
        for (int t = 1; t < length_of_simulations; t += 1) {
            double draw = distribution(generator);
            for (int j = 0; j < m_number_of_states; j += 1) {
                if (cdfP(state_vec_idx(t-1),j) > draw) {
                    state_vec_idx(t) = j;
                    state_vec(t) = m_S(state_vec_idx(t));
                    break;
                }
            }
        }
        simulations.emplace_back(SimDTMC(state_vec, state_vec_idx));
    }
    return simulations;
}


