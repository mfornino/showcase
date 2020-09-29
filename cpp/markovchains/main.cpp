//
// Created by Michele Fornino on 5/5/18.

// This is the base code for the MarkovChains project. It provides a few examples of instantiations and usage of the
// classes MarkovChain.h and MarkovChainSimulations.h
//

//#include "DenseDTMC.h"
//#include "DenseCTMC.h"
//#include "SparseCTMC.h"
//#include "SimCTMC.h"

#include "MarkovChain.h"
#include "MarkovChainSimulations.h"
#include <iostream>
//#include <vector>
//#include <Eigen/Dense>
//#include <Eigen/Sparse>
//#include <Eigen/SparseLU>

using namespace Eigen;
using namespace std;
using namespace MarkovChain;
using namespace MarkovChainSimulations;

int main() {



//    typedef SparseMatrix<double, RowMajor> SMr;
//    typedef Triplet<double> T;
    //// Dense Discrete Time Markov Chain
/*
    // Populate Transition Matrix from Rouwenhorst convenience function
    DenseDTMC chain = DenseDTMC::discretizeAR1_Rouwenhorst();

    // Populate DenseDTMC object and compute sample path
    DenseDTMC::ManySimDTMC simulations;
    simulations = chain.simDenseDTMC();
*/
    DenseCTMC chain1; chain1.discretizeAR1_Rouwenhorst();
//    DenseDTMC chain2; chain2.discretizeAR1_Rouwenhorst();
    SimCTMC simulation1 = chain1.simulateMarkovChain(2);
//    SimDTMC simulation2 = chain2.simulateMarkovChain(2);

//    cout << chain1;
//    cout << chain2;
//    cout << simulation1;
//    cout << simulation2;


//#pragma omp declare reduction (merge : std::vector<SimCTMC> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    MultSimCTMC prova;
//#pragma omp parallel for reduction(merge: prova)
    for (int k = 0; k < 80; k += 1){
        prova.emplace_back(chain1.simulateMarkovChain(2, 10000));
    }

    cout << "DONE SIMULATING!!\n";

    cout << prova[79];

//    MultSimDTMC prova; prova.emplace_back(simulation2); prova.emplace_back(simulation2);
//    saveToFile(prova);

//    simulation2.saveToFile();
//    MarkovChain<ContinuousTime, MatrixXd> chain = MarkovChain<Continuous, MatrixXd>::discretizeAR1_Rouwenhorst();

    //// Sparse Continuous Time Markov Chain
    // Populate Infinitesimal Generator
//    SMr Q(4,4);
//    vector<T> tripletQ; tripletQ.reserve(10);
//    tripletQ.emplace_back(T(0,0,-0.25));
//    tripletQ.emplace_back(T(0,1, 0.25));
//    tripletQ.emplace_back(T(1,0, 0.15));
//    tripletQ.emplace_back(T(1,1,-0.50));
//    tripletQ.emplace_back(T(1,2, 0.35));
//    tripletQ.emplace_back(T(3,3,-0.25));
//    tripletQ.emplace_back(T(3,2, 0.25));
//    tripletQ.emplace_back(T(2,1, 0.25));
//    tripletQ.emplace_back(T(2,2,-0.50));
//    tripletQ.emplace_back(T(2,3, 0.25));
//    Q.setFromTriplets(tripletQ.begin(), tripletQ.end());
//
//    // Populate State Vector (Eigen vector of double)
//    VectorXd S(4); S(0) = -0.5; S(1) = 0; S(2) = 0.5; S(3) = 1;

    // Populate SparseCTMC object and compute sample path
//    SparseCTMC chain = SparseCTMC(Q, S);
//    SparseCTMC::ManySimCTMC simulations = chain.simSparseCTMC();


    //// Dense Continuous Time Markov Chain
/*
    // Populate Infinitesimal Generator
    MatrixXd Q(4,4);
    Q(0,0) = -0.03; Q(0,1) =  0.02; Q(0,2) =  0.01; Q(0,3) =  0.00;
    Q(1,0) =  0.10; Q(1,1) = -0.55; Q(1,2) =  0.30; Q(1,3) =  0.15;
    Q(2,0) =  0.10; Q(2,1) =  0.25; Q(2,2) = -0.60; Q(2,3) =  0.25;
    Q(3,0) =  0.10; Q(3,1) =  0.15; Q(3,2) =  0.30; Q(3,3) = -0.55;

    // Populate State Vector (Eigen vector of double)
    VectorXd S(4); S(0) = -0.5; S(1) = 0; S(2) = 0.5; S(3) = 1;

    // Populate DenseCTMC object and compute sample path
    DenseCTMC chain = DenseCTMC(Q, S);
    DenseCTMC::ManySimCTMC simulations = chain.simDenseCTMC();
*/

    //// Display Results
//    cout << "Stationary Distribution:\n" << chain.getStationaryDistribution() << "\n";
//    cout << "Simulation:\n\n" << simulations[0];

    return 0;
}