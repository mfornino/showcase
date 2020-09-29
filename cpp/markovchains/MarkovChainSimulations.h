//
// Created by Michele Fornino on 5/12/18.
//

#ifndef MARKOVCHAINS_MARKOVCHAINSIMULATIONS_H
#define MARKOVCHAINS_MARKOVCHAINSIMULATIONS_H

// Baseline includes
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>

// Eigen includes
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>

using namespace Eigen;
using namespace std;

const string MARKOVCHAINSIMULATIONS_RELATIVE_PATH = "../user/";

namespace MarkovChainSimulations {

    //// Superclass
    class SimMC {
    protected:
        VectorXd m_state_vec;
        VectorXi m_state_vec_idx;
        VectorXi m_draw_idx;
        long int m_chain_length;


    public:

        //// Constructors
        SimMC() = default;
        ~SimMC() = default;
        SimMC( VectorXd &state_vec, VectorXi &state_vec_idx){
            long int chain_length = state_vec.rows();
            setSimMC(state_vec, state_vec_idx, chain_length);
        };
        SimMC( VectorXd &state_vec, VectorXi &state_vec_idx, long int &chain_length){
            setSimMC(state_vec, state_vec_idx, chain_length);
        };
        void setSimMC( VectorXd &state_vec, VectorXi &state_vec_idx, long int &chain_length){
            m_state_vec = state_vec;
            m_state_vec_idx = state_vec_idx;
            m_chain_length = chain_length;
            m_draw_idx = VectorXi::LinSpaced(chain_length, 0, static_cast<int>(chain_length - 1));
        };

        //// Methods to obtain objects
        VectorXi getSimDraw() { return m_draw_idx; }
        VectorXd getSimState(){ return m_state_vec; }
        VectorXi getSimStateIndex(){ return m_state_vec_idx; }
        long int getSimChainLength(){ return m_chain_length; }
    };

    //// Child: Discrete Time Markov Chains
    class SimDTMC: public SimMC {
    private:
        VectorXd m_time;

    public:

        //// Constructors
        SimDTMC() = default;
        ~SimDTMC() = default;
        SimDTMC(VectorXd &state_vec, VectorXi &state_vec_idx)
                : SimMC(state_vec, state_vec_idx)
        {
            m_time = m_draw_idx.cast<double>();
        };
        SimDTMC(VectorXd &state_vec, VectorXi &state_vec_idx, double &time_frequency)
                : SimMC(state_vec, state_vec_idx)
        {
            m_time = VectorXd::LinSpaced(m_chain_length, 0, (m_chain_length - 1) / time_frequency);
        };
        SimDTMC(VectorXd &state_vec, VectorXi &state_vec_idx, long int &chain_length)
                : SimMC(state_vec, state_vec_idx, chain_length)
        {
            m_time = m_draw_idx.cast<double>();
        };
        SimDTMC(VectorXd &state_vec, VectorXi &state_vec_idx, long int &chain_length, double &time_frequency)
                : SimMC(state_vec, state_vec_idx, chain_length)
        {
            m_time = VectorXd::LinSpaced(chain_length, 0, (chain_length - 1) / time_frequency);
        };

        //// Methods to obtain objects
        VectorXd getSimTime(){ return m_time; }

        //// Method to export simulation as CSV file
        void saveToFile(string filename = "simulation.csv");

        //// Add operator overloading for << (cout)
        friend ostream & operator << (ostream &out, SimDTMC &c) {
            int col = 20;

            cout << "Discrete Time Markov Chain Simulation.\n";
            cout << "Chain Length: " << c.m_chain_length << "\n";
            if (c.m_chain_length > 0) {
                cout << left << setw(col) << "Draw"
                     << left << setw(col) << "Time"
                     << left << setw(col) << "State"
                     << left << setw(col) << "State (index)"
                     << endl;
                for (int t = 0; t < c.m_chain_length; t += 1){
                    cout << left << setw(col) << c.m_draw_idx(t)
                         << left << setw(col) << c.m_time(t)
                         << left << setw(col) << c.m_state_vec(t)
                         << left << setw(col) << c.m_state_vec_idx(t)
                         << endl;
                }
            } else {
                cout << "NO DATA STORED";
            }
            return out;
        };
    };

    //// Child: Continuous Time Markov Chains
    class SimCTMC: public SimMC {
    private:
        VectorXd m_time;
        VectorXd m_arrival_times;

        //// Add Constructor for this case (takes care of the arrival times)

        public:

        //// Constructors
        SimCTMC() = default;
        ~SimCTMC() = default;
        SimCTMC(VectorXd &state_vec, VectorXi &state_vec_idx, VectorXd &arrival_times)
        : SimMC(state_vec, state_vec_idx)
        {
            setSimCTMCTime(arrival_times);
        };
        SimCTMC(VectorXd &state_vec, VectorXi &state_vec_idx, VectorXd &arrival_times, long int &chain_length)
        : SimMC(state_vec, state_vec_idx, chain_length)
        {
            setSimCTMCTime(arrival_times);
        };
        void setSimCTMCTime(VectorXd &arrival_times){
            m_arrival_times = arrival_times;
            VectorXd time(m_chain_length + 1); time(0) = 0;
            for (int i = 1; i < m_chain_length; i += 1){
                time(i) = time(i-1) + arrival_times(i-1);
            };
            m_time = time;
        };

        //// Methods to obtain objects
        VectorXd getSimTime(){ return m_time; }
        VectorXd getSimArrivalTimes(){ return m_arrival_times; }

        //// Method to export simulation as CSV file
        void saveToFile(string filename = "simulation.csv");

        //// Add operator overloading for << (cout)
        friend ostream & operator << (ostream &out, SimCTMC &c) {
            int col = 20;

            cout << "Continuous Time Markov Chain Simulation.\n";
            cout << "Chain Length: " << c.m_chain_length << "\n";
            if (c.m_chain_length > 0) {
                cout << left << setw(col) << "Draw"
                     << left << setw(col) << "Time"
                     << left << setw(col) << "Arrival Time"
                     << left << setw(col) << "State"
                     << left << setw(col) << "State (index)"
                     << endl;
                for (int t = 0; t < c.m_chain_length; t += 1){
                    cout << left << setw(col) << c.m_draw_idx(t)
                         << left << setw(col) << c.m_time(t)
                         << left << setw(col) << c.m_arrival_times(t)
                         << left << setw(col) << c.m_state_vec(t)
                         << left << setw(col) << c.m_state_vec_idx(t)
                         << endl;
                }
            } else {
                cout << "NO DATA STORED";
            }
            return out;
        };
    };

    void SimCTMC::saveToFile(string filename) {

        // Declare File Name + Path and create folder
        time_t rawtime;
        struct tm * timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        char buff[20] = "2018-01-01_00-00-00"; // REMEMBER THE FUCKING NULL TERMINATOR CHARACTER :)
        strftime(buff, 20, "%Y-%m-%d_%H-%M-%S", timeinfo);
        string pathToFile = MARKOVCHAINSIMULATIONS_RELATIVE_PATH + "mcsimulations/" + static_cast<string>(buff) + "/";
        string command = "mkdir -p " + pathToFile;
        system(command.c_str());

        // Open stream
        ofstream outf(pathToFile + filename, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        outf << "draw,time,arrivaltime,state,stateidx" << endl;
        for (int i = 0; i < m_chain_length; i += 1){
            outf << m_draw_idx(i) << "," << m_time(i) << "," << m_arrival_times(i) << "," << m_state_vec(i) << "," << m_state_vec_idx(i) << endl;
        }

        // Restore precision
        outf.precision(ss);
    }

    void SimDTMC::saveToFile(string filename) {

        // Declare File Name + Path and create folder
        time_t rawtime;
        struct tm * timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        char buff[20] = "2018-01-01_00-00-00"; // REMEMBER THE FUCKING NULL TERMINATOR CHARACTER :)
        strftime(buff, 20, "%Y-%m-%d_%H-%M-%S", timeinfo);
        string pathToFile = MARKOVCHAINSIMULATIONS_RELATIVE_PATH + "mcsimulations/" + static_cast<string>(buff) + "/";
        string command = "mkdir -p " + pathToFile;
        system(command.c_str());

        // Open stream
        ofstream outf(pathToFile + filename, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        outf << "draw,time,state,stateidx" << endl;
        for (int i = 0; i < m_chain_length; i += 1){
            outf << m_draw_idx(i) << "," << m_time(i) << "," << m_state_vec(i) << "," << m_state_vec_idx(i) << endl;
        }

        // Restore precision
        outf.precision(ss);
    }


    typedef vector<SimDTMC> MultSimDTMC;
    void saveToFile(MultSimDTMC &simulations, string filename = "simulations.csv"){
        // Declare File Name + Path
        time_t rawtime;
        struct tm * timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        char buff[20] = "2018-01-01_00-00-00"; // REMEMBER THE FUCKING NULL TERMINATOR CHARACTER :)
        strftime(buff, sizeof buff, "%Y-%m-%d_%H-%M-%S", timeinfo);
        string pathToFile = MARKOVCHAINSIMULATIONS_RELATIVE_PATH + "mcsimulations/" + static_cast<string>(buff) + "/";
        string command = "mkdir -p " + pathToFile;
        system(command.c_str());

        // Open stream
        ofstream outf(pathToFile + filename, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        outf << "simulation,draw,time,state,stateidx" << endl;
        for (int k = 0; k < simulations.size(); k += 1){
            for (int i = 0; i < simulations[k].getSimChainLength(); i += 1){
                outf << k << ","
                     << simulations[k].getSimDraw()(i) << ","
                     << simulations[k].getSimTime()(i) << ","
                     << simulations[k].getSimState()(i) << ","
                     << simulations[k].getSimStateIndex()(i) << endl;
            }
        }

    };

    typedef vector<SimCTMC> MultSimCTMC;
    void saveToFile(MultSimCTMC &simulations, string filename = "simulations.csv"){
        // Declare File Name + Path
        time_t rawtime;
        struct tm * timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        char buff[20] = "2018-01-01_00-00-00"; // REMEMBER THE FUCKING NULL TERMINATOR CHARACTER :)
        strftime(buff, 20, "%Y-%m-%d_%H-%M-%S", timeinfo);
        string pathToFile = MARKOVCHAINSIMULATIONS_RELATIVE_PATH + "mcsimulations/" + static_cast<string>(buff) + "/";
        string command = "mkdir -p " + pathToFile;
        system(command.c_str());

        // Open stream
        ofstream outf(pathToFile + filename, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        outf << "simulation,draw,time,arrivaltime,state,stateidx" << endl;
        for (int k = 0; k < simulations.size(); k += 1){
            for (int i = 0; i < simulations[k].getSimChainLength(); i += 1){
                outf << k << ","
                     << simulations[k].getSimDraw()(i) << ","
                     << simulations[k].getSimTime()(i) << ","
                     << simulations[k].getSimArrivalTimes()(i) << ","
                     << simulations[k].getSimState()(i) << ","
                     << simulations[k].getSimStateIndex()(i) << endl;
            }
        }

    };

}; // Close Namespace
#endif



