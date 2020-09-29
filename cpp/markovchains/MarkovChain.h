//
// Created by Michele Fornino on 5/12/18.
//

#ifndef MARKOVCHAINS_MARKOVCHAIN_H
#define MARKOVCHAINS_MARKOVCHAIN_H

// Baseline includes
#include <random>
#include <iostream>

// MarkovChainSimulations needed for simulation methods of the classes (it represents the output)
#include "MarkovChainSimulations.h"

// Eigen includes
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>

using namespace Eigen;
using namespace std;

namespace MarkovChain {

    //// Definition of Time Type of Markov Chain
    enum MCType {
        DiscreteTime,
        ContinuousTime,
    };

    //// MarkovChain Class Template
    template<MCType TimeType = DiscreteTime, class Storage = MatrixXd>
    class MarkovChain {

    protected:
        Storage m_matrix;
        long m_size_of_state_space;
        VectorXd m_state_space;

    public:

        //// Standard direct constructors
        MarkovChain() = default;
        ~MarkovChain() = default;
        explicit MarkovChain(Storage &Q) {
            VectorXd S; S = VectorXd::LinSpaced(Q.rows(),0,Q.rows());
            setMarkovChain(Q, S);
        };
        MarkovChain(Storage &Q, const VectorXd &S) {
            setMarkovChain(Q, S);
        };
        void setMarkovChain(Storage &Q, const VectorXd &S) {
            m_matrix = Q;
            m_state_space = S;
            m_size_of_state_space = Q.rows();
        };

        //// Accessors to private members
        MatrixXd getTransitionMatrix() { return m_matrix; }
        VectorXd getStateSpace() { return m_state_space; }
        double getStateSpaceSize() { return m_size_of_state_space; }

    };

    //// Children of SuperClass: Continuous Time Markov Chain
    class DenseCTMC: public MarkovChain<ContinuousTime, MatrixXd> {
    public:
        //// Constructors
        DenseCTMC() : MarkovChain() {};
        explicit DenseCTMC(MatrixXd &Q) : MarkovChain(Q) {};
        DenseCTMC(MatrixXd &Q, const VectorXd &S) : MarkovChain(Q,S) {};

        //// Methods
        void discretizeAR1_Rouwenhorst(double rho = 0.8, double sigma_eps = 0.1, double mu = 0, int number_of_states = 9);
        VectorXd getStationaryDistribution(double step_size = 1e3);
        MarkovChainSimulations::SimCTMC simulateMarkovChain(int starting_point_index = 0, long int chain_length = 1000);

        //// Add operator overloading for << (cout)
        friend ostream & operator << (ostream &out, const DenseCTMC &c) {
            cout << "\nContinuous Time Markov Chain with " << c.m_size_of_state_space << " states.\n\n";
            cout << "Infinitesimal Generator:\n";
            cout << c.m_matrix << "\n\n";
            cout << "State Space:\n";
            cout << c.m_state_space << "\n";
            return out;
        };
    };

    class SparseCTMC: public MarkovChain<ContinuousTime, SparseMatrix<double, RowMajor>> {
    public:
        //// Constructors
        SparseCTMC() : MarkovChain() {};
        explicit SparseCTMC(SparseMatrix<double, RowMajor> &Q) : MarkovChain(Q) {};
        SparseCTMC(SparseMatrix<double, RowMajor> &Q, const VectorXd &S) : MarkovChain(Q,S) {};

        //// Methods
        VectorXd getStationaryDistribution(double step_size = 1e3);
        MarkovChainSimulations::SimCTMC simulateMarkovChain(int starting_point_index = 0, long int chain_length = 1000);

        //// Add operator overloading for << (cout)
        friend ostream & operator << (ostream &out, const SparseCTMC &c) {
            cout << "\nContinuous Time Markov Chain with " << c.m_size_of_state_space << " states.\n\n";
            cout << "Infinitesimal Generator:\n";
            cout << c.m_matrix << "\n\n";
            cout << "State Space:\n";
            cout << c.m_state_space << "\n";
            return out;
        };
    };


    //// Children of SuperClass: Discrete Time Markov Chain
    class DenseDTMC: public MarkovChain<DiscreteTime, MatrixXd> {
    public:
        //// Constructors
        DenseDTMC() : MarkovChain() {};
        explicit DenseDTMC(MatrixXd &Q) : MarkovChain(Q) {};
        DenseDTMC(MatrixXd &Q, const VectorXd &S) : MarkovChain(Q,S) {};

        //// Methods
        void discretizeAR1_Rouwenhorst(double rho = 0.8, double sigma_eps = 0.1, double mu = 0, int number_of_states = 9);
        VectorXd getStationaryDistribution(bool iterative = true, double criterion_iterative = 1e-10, int maxit = 1000);
        MarkovChainSimulations::SimDTMC simulateMarkovChain(int starting_point_index = 0, long int chain_length = 1000);

        //// Add operator overloading for << (cout)
        friend ostream & operator << (ostream &out, const DenseDTMC &c) {
            cout << "\nDiscrete Time Markov Chain with " << c.m_size_of_state_space << " states.\n\n";
            cout << "Transition Matrix:\n";
            cout << c.m_matrix << "\n\n";
            cout << "State Space:\n";
            cout << c.m_state_space << "\n";
            return out;
        };
    };



    //// Function Declarations
    void DenseDTMC::discretizeAR1_Rouwenhorst(double rho, double sigma_eps, double mu, int number_of_states){
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
            MatrixXd tm(2, 2);
            tm(0, 0) = p;
            tm(0, 1) = 1 - p;
            tm(1, 0) = 1 - q;
            tm(1, 1) = q;
            for (int i = 2; i < number_of_states; i += 1) {
                MatrixXd newtm = MatrixXd::Zero(i + 1, i + 1);
                newtm.topLeftCorner(i, i) += p * tm;
                newtm.topRightCorner(i, i) += (1.0 - p) * tm;
                newtm.bottomLeftCorner(i, i) += (1.0 - q) * tm;
                newtm.bottomRightCorner(i, i) += q * tm;
                newtm.block(1, 0, i - 1, i + 1) = newtm.block(1, 0, i - 1, i + 1) / 2.0;
                tm = newtm;
            }

            // Construct Appropriate State Space
            double psi = sqrt(number_of_states - 1) * sigma_eps / sqrt(1 - pow(rho, 2.0));
            VectorXd S = VectorXd::LinSpaced(number_of_states, mu / (1.0 - rho) - psi, mu / (1.0 - rho) + psi);

            MarkovChain::setMarkovChain(tm,S);

        } else {

            // Handles case with number of states = 1
            MatrixXd tm(1, 1);
            VectorXd S(1);
            tm(0, 0) = 1;
            S(0) = 1;
            MarkovChain::setMarkovChain(tm,S);
        };
    }

    void DenseCTMC::discretizeAR1_Rouwenhorst(double rho, double sigma_eps, double mu, int number_of_states){

        // Get transition matrix of embedded Markov Chain
        DenseDTMC temp; temp.discretizeAR1_Rouwenhorst(rho,sigma_eps,mu,number_of_states);
        MatrixXd Q = temp.getTransitionMatrix();

        // Translate it into the corresponding infinitesimal generator
        // Remark: handles the case with 1 element just fine.
        for (int i = 0; i < temp.getStateSpaceSize(); i += 1){
            Q(i,i) = 0;
            double sum = Q.row(i).sum();
            Q(i,i) = -sum;
        }

        // Set the MarkovChain object
        setMarkovChain(Q, temp.getStateSpace());

    };


    //// Stationary Distribution
    VectorXd DenseCTMC::getStationaryDistribution(double step_size){
        // This Method Solves the KF Equations, a.k.a. the Fokker-Planck Equations for the CTMC.

        // Initialize distribution
        VectorXd distribution = VectorXd::Constant(m_size_of_state_space, 1.0/m_size_of_state_space);
        MatrixXd eye = MatrixXd::Identity(m_size_of_state_space,m_size_of_state_space);
        MatrixXd A(m_size_of_state_space,m_size_of_state_space);
        A = eye - step_size * m_matrix.transpose();

        // Solve System for Stationary Distribution
        return A.partialPivLu().solve(distribution);
    };

    VectorXd SparseCTMC::getStationaryDistribution(double step_size){

        // This Method Solves the KF Equations, a.k.a. the Fokker-Planck Equations for the Sparse CTMC.

        // Initialize distribution
        VectorXd distribution = VectorXd::Constant(m_size_of_state_space, 1.0/m_size_of_state_space);
        SparseMatrix<double, ColMajor> eye(m_size_of_state_space, m_size_of_state_space);
        eye.setIdentity();
        SparseMatrix<double, ColMajor> A(m_size_of_state_space,m_size_of_state_space);
        A = eye - step_size * m_matrix.transpose();

        // Define Solver: SparseLU
        SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);

//        // BiCGSTAB
//        BiCGSTAB<SparseMatrix<double, ColMajor>> solver;
//        solver.setTolerance(1e-16);
//        solver.compute(A);

        return solver.solve(distribution);
    };

    VectorXd DenseDTMC::getStationaryDistribution(bool iterative, double criterion_iterative, int maxit){
        // This Method Solves for the stationary distribution.
        // Initialize
        VectorXd p = VectorXd::Constant(m_size_of_state_space,1.0/m_size_of_state_space);

        if (!iterative) {
//
//        //TODO ADD THIS!!!
            VectorXd identity_distribution = VectorXd::Constant(m_size_of_state_space, 1.0/m_size_of_state_space);
            return identity_distribution;

        } else if (iterative) {
            MatrixXd PT = m_matrix.transpose();
            VectorXd p_next;
            double supnorm = 1e10;
            double it = 0;
            while(supnorm > criterion_iterative && it < maxit){
                p_next = PT * p;
                double maxabsdiff = -1.0;
                double absdiff;
                for (int i = 0; i < m_size_of_state_space; i += 1){
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
    };



    //// Montecarlo Simulations
    MarkovChainSimulations::SimDTMC DenseDTMC::simulateMarkovChain(int starting_point_index, long int chain_length){
        // Define Random Engine
        std::mt19937_64 generator;

        // Allocate output structure
        MarkovChainSimulations::SimDTMC simulation;

        // Run simulations. Use Skorohod theorem.
        // Compute CDF of each conditional transition probability function
        MatrixXd cdfP(m_size_of_state_space,m_size_of_state_space);
        for (int j = 0; j < m_size_of_state_space; j += 1) {
            cdfP.col(j) = m_matrix.leftCols(j + 1).rowwise().sum();
        }

        // Initialize uniform distribution over [0,1]
        uniform_real_distribution<double> distribution(0.0,1.0);

        // Compute chain
        VectorXd state_vec(chain_length);
        VectorXi state_vec_idx(chain_length);
        state_vec_idx(0) = starting_point_index;
        state_vec(0) = m_state_space(state_vec_idx(0));
        for (long int t = 1; t < chain_length; t += 1) {
            double draw = distribution(generator);
            for (int j = 0; j < m_size_of_state_space; j += 1) {
                if (cdfP(state_vec_idx(t-1),j) > draw) {
                    state_vec_idx(t) = j;
                    state_vec(t) = m_state_space(state_vec_idx(t));
                    break;
                }
            }
        }

        // Construct MarkovChainSimulations object
        return MarkovChainSimulations::SimDTMC(state_vec, state_vec_idx, chain_length);
    };

    MarkovChainSimulations::SimCTMC DenseCTMC::simulateMarkovChain(int starting_point_index, long int chain_length){
        // Define Random Engine
        std::mt19937_64 generator;

        // Allocate output structure
        MarkovChainSimulations::SimCTMC simulation;

        // Run simulations
//        double tol = 1e-6;
        VectorXd state_vec(chain_length);
        VectorXi state_vec_idx(chain_length);
        VectorXd arrival_times_vec(chain_length);
        state_vec_idx(0) = starting_point_index;
        state_vec(0) = m_state_space(starting_point_index);
        for (long int t = 1; t < chain_length; t += 1) {
            double mindraw = 1e10;
            int minidx = 0;
            for(int i = 0; i < m_size_of_state_space; i += 1) {
                if (i != state_vec_idx(t-1)) {
                    double rate = m_matrix( state_vec_idx(t - 1), i);
                    exponential_distribution<double> distribution(rate);
                    double draw = distribution(generator);
                    if (draw < mindraw) {
                        mindraw = draw;
                        minidx = i;
                    }
                }
            }
            state_vec_idx(t) = minidx;
            state_vec(t) = m_state_space(minidx);
            arrival_times_vec(t - 1) = mindraw;
        }

        return MarkovChainSimulations::SimCTMC(state_vec, state_vec_idx, arrival_times_vec, chain_length);
    };

    MarkovChainSimulations::SimCTMC SparseCTMC::simulateMarkovChain(int starting_point_index, long int chain_length){
        // Define Random Engine
        std::default_random_engine generator;

        // Allocate output structure
        MarkovChainSimulations::SimCTMC simulation;

        // Run simulations

        VectorXd state_vec(chain_length);
        VectorXi state_vec_idx(chain_length);
        state_vec_idx(0) = starting_point_index;
        state_vec(0) = m_state_space(state_vec_idx(0));
        VectorXd arrival_times_vec(chain_length);
        for (long int t = 1; t < chain_length; t += 1) {
            double mindraw = 1e10;
            int minidx = 0;
            double draw;
            for (SparseMatrix<double, RowMajor>::InnerIterator it(m_matrix,state_vec_idx(t - 1)); it; ++it)
            {
                if (it.col() != it.row()) {
                    exponential_distribution<double> distribution(it.value());
                    draw = distribution(generator);
                    if (draw < mindraw) {
                        mindraw = draw;
                        minidx = it.col();
                    }
                }
            }
            state_vec_idx(t) = minidx;
            state_vec(t) = m_state_space(minidx);
            arrival_times_vec(t - 1) = mindraw;
        }


        return MarkovChainSimulations::SimCTMC(state_vec, state_vec_idx, arrival_times_vec, chain_length);
    };


}; // Close namespace

#endif
