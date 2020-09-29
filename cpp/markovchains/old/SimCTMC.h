//
// Created by Michele Fornino on 5/12/18.
//

#ifndef MARKOVCHAINS_SIMCTMC_H
#define MARKOVCHAINS_SIMCTMC_H

#include <iomanip>
#include <iostream>
#include <eigen/3.3.4/include/eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;

class SimCTMC {
private:
    VectorXd m_arrival_time_vec;
    VectorXd m_time_vec;
    VectorXd m_state_vec;
    VectorXi m_state_vec_idx;
    long m_number_of_draws_per_simulation;

public:

    // Explicitly add default constructor
    SimCTMC() = default;

    // Constructor
    SimCTMC(VectorXd arrival_time_vec, VectorXd state_vec, VectorXi state_vec_idx)
    {
        m_number_of_draws_per_simulation = arrival_time_vec.rows();
        m_arrival_time_vec = arrival_time_vec;
        VectorXd time(m_number_of_draws_per_simulation + 1);
        for (int t = 1; t <= m_number_of_draws_per_simulation; t += 1){
            time(t) = time(t-1) + m_arrival_time_vec(t-1);
        }
        m_time_vec = time;
        m_state_vec = state_vec;
        m_state_vec_idx = state_vec_idx;
    };

    // Add methods to save to file
    // TODO

    // Add operator overloading for << (cout)
    friend ostream & operator << (ostream &out, const SimCTMC &c) {
        double col = 20;

        if (c.m_number_of_draws_per_simulation > 0) {
            cout << left << setw(col) << "TIME"
                 << left << setw(col) << "ARRIVAL TIME"
                 << left << setw(col) << "STATE"
                 << left << setw(col) << "INDEX OF STATE"
                 << endl;
            for (int t = 0; t < c.m_number_of_draws_per_simulation; t += 1){
                cout << left << setw(col) << c.m_time_vec(t)
                     << left << setw(col) << c.m_arrival_time_vec(t)
                     << left << setw(col) << c.m_state_vec(t)
                     << left << setw(col) << c.m_state_vec_idx(t)
                     << endl;
            }
        } else {
            cout << "NO DATA STORED";
        }


        return out;
    };

    // Add methods to make graph
    // TODO
};

#endif
