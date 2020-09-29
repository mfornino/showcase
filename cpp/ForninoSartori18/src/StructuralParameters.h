//
// Created by Michele Fornino on 5/11/18.
//

#include <fstream>
#include <iomanip>
#include <iostream>
#include "markovchains/MarkovChain.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#ifndef STRUCTURALPARAMETERS_H
#define STRUCTURALPARAMETERS_H

using namespace std;
using namespace MarkovChain;

const string STRUCTURALPARAMETERS_RELATIVE_PATH = "../user/";


// Parameters of parameters' file.
const int sp_col = 40;         // Set size of column in parameters file
const int sp_start = 3;        // start from line 4
const int sp_max_lines = 100;  // read through line 100

// Defaults defined as global for convenience.
const int default_nz = 4;
const int default_nbar = 60;
const double default_wl = 1.0;
const double default_wh = 1.25;
const double default_decreasing_returns = 2.0/3.0;
const double default_real_rate = 0.01;
const double default_fixed_cost = 1.5;
const double default_exit_rate = 1e5;
const double default_layoff_rate = 13.0;
const double default_layoff_cost = 1.0;
const double default_exogenous_separations_rate = 1.0/16.0;
const double default_aggregate_productivity = 1.0;
const double default_rho = 0.8;
const double default_sigma = 0.1;
const double default_mu = 0.55;
const double default_disaster_rate = 0.1;
const Vector4d default_distribution_of_entrants = {0.00, 0.25, 0.50, 0.25};

class StructuralParameters
{

public:

    //// Member Parameters
    // State space
    int nz;
    int nbar;
    double wl;
    double wh;

    // Deep parameters
    double decreasing_returns;
    double real_rate;
    double discount_rate;
    double fixed_cost;
    double exit_rate;
    double layoff_rate;
    double layoff_cost;
    double exogenous_separations_rate;
    double aggregate_productivity;

    // Parameters of the underlying productivity process outside financial disaster
    double rho;
    double sigma;
    double mu;
    double disaster_rate;
    MatrixXd Q;
    VectorXd productivity;

    // Distribution of entrants
    VectorXd distribution_of_entrants;

    // Embedded Markov Chain of the Poisson shock
//    DenseDTMC embeddedMarkovChain;

    //// Constructor to initialize reading appropriately defined file
    StructuralParameters() {
        setDefaults();
    }


    explicit StructuralParameters(string &filename)
    {
        loadFromFile(filename);
    };

    // setDefaults Constructor helper
    void setDefaults() {
        // Set member values
        nz = default_nz;
        nbar = default_nbar;
        wl = default_wl;
        wh = default_wh;
        decreasing_returns = default_decreasing_returns;
        real_rate = default_real_rate;
        fixed_cost = default_fixed_cost;
        exit_rate = default_exit_rate;
        layoff_rate = default_layoff_rate;
        layoff_cost = default_layoff_cost;
        exogenous_separations_rate = default_exogenous_separations_rate;
        aggregate_productivity = default_aggregate_productivity;
        rho = default_rho;
        sigma = default_sigma;
        mu = default_mu;
        disaster_rate = default_disaster_rate;
        distribution_of_entrants = default_distribution_of_entrants;

        // Set derived member values
        setDerived();
    }

    // loadFromFile Constructor Helper
    void loadFromFile(string &filename) {
        // Declare File Name
        string pathToFile = STRUCTURALPARAMETERS_RELATIVE_PATH + filename;
        ifstream inf(pathToFile, ios::binary);

        if (!inf) {

            cout << "Could not load from file. Setting defaults.";
            setDefaults();

        } else {

            // Parse file. First grab
            string str;
            vector<double> inputs;
            int line = 0;
            while (line < sp_max_lines) {
                getline(inf, str);
                if (line >= sp_start && !str.empty()) {
                    inputs.emplace_back(atof(str.substr(40).c_str()));
                };
                line += 1;
            }

            // Set member values
            nz = (int)round(inputs[0]);
            nbar = (int)round(inputs[1]);
            wl = inputs[2];
            wh = inputs[3];
            decreasing_returns = inputs[4];
            real_rate = inputs[5];
            fixed_cost = inputs[6];
            exit_rate = inputs[7];
            layoff_rate = inputs[8];
            layoff_cost = inputs[9];
            exogenous_separations_rate = inputs[10];
            aggregate_productivity = inputs[11];
            rho = inputs[12];
            sigma = inputs[13];
            mu = inputs[14];
            disaster_rate = inputs[15];
            distribution_of_entrants = VectorXd::Constant(nz,0);
            for (int i = 16; i < 16+nz; i += 1) {
                distribution_of_entrants(i-16) = inputs[i];
            }

            // Set derived member values
            setDerived();
        }
    };

    // setDerived Constructor helper
    void setDerived() {
        discount_rate = log(1.0 + real_rate);

//         Create Infinitesimal Generator of Poisson Process for Productivity
        DenseCTMC ig; ig.discretizeAR1_Rouwenhorst(rho, sigma, mu, nz - 1);
        MatrixXd TM = ig.getTransitionMatrix();
        for (int i = 0; i < nz - 1; i += 1){
            double factor = 1.0/TM(i,i);
            TM(i,i) += - disaster_rate;
            TM.row(i) = TM.row(i) * factor * TM(i,i);    // I want the total out rate to be just like before.
        }
        MatrixXd newTM(nz,nz);
        newTM.bottomRightCorner(nz-1,nz-1) = TM;
        newTM.row(0) = VectorXd::Constant(nz, 0);
        newTM.bottomLeftCorner(nz-1, 1) = VectorXd::Constant(nz-1, disaster_rate);
        Q = newTM;

        VectorXd aux = ig.getStateSpace();
        VectorXd Z; Z = VectorXd::Constant(nz,0); Z.tail(nz-1) = aux;
        for (int i = 1; i < nz; i += 1){
            Z(i) = aux(i-1);
        }
        productivity = Z;
    }

    //// Overloading stream insertion operators
    friend ostream & operator << (ostream &out, const StructuralParameters &c);
    friend istream & operator >> (istream &in,  StructuralParameters &c);

    //// Methods
    void saveToFile(string filename = "structuralParameters.txt") {

        // Declare File Name
        string pathToFile = STRUCTURALPARAMETERS_RELATIVE_PATH + filename;
        system("mkdir -p ../user/");
        ofstream outf(pathToFile, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        // Save Values
        outf << left << "STRUCTURAL PARAMETERS:" << endl << endl;
        outf << left << setw(sp_col) << "NAME" << "VALUE" << endl;
        outf << left << setw(sp_col) << "Number of productivity shocks: "      << nz << endl;
        outf << left << setw(sp_col) << "Maximum firm size: "                  << nbar << endl;
        outf << left << setw(sp_col) << "Low Wage: "                           << wl << endl;
        outf << left << setw(sp_col) << "High Wage: "                          << wh << endl;
        outf << left << setw(sp_col) << "Decreasing returns parameter: "       << decreasing_returns << endl;
        outf << left << setw(sp_col) << "Real interest rate: "                 << real_rate << endl;
        outf << left << setw(sp_col) << "Instantaneous fixed cost: "           << fixed_cost << endl;
        outf << left << setw(sp_col) << "Exit rate: "                          << exit_rate << endl;
        outf << left << setw(sp_col) << "Layoff rate: "                        << layoff_rate << endl;
        outf << left << setw(sp_col) << "Instantaneous severance pay: "        << layoff_cost << endl;
        outf << left << setw(sp_col) << "Exogenous separation rate: "          << exogenous_separations_rate << endl;
        outf << left << setw(sp_col) << "Aggregate productivity shifter: "     << aggregate_productivity << endl;
        outf << left << setw(sp_col) << "AR(1) rho: "                          << rho << endl;
        outf << left << setw(sp_col) << "AR(1) sigma_epsilon: "                << sigma << endl;
        outf << left << setw(sp_col) << "AR(1) mu: "                           << mu << endl;
        outf << left << setw(sp_col) << "Financial Disaster Rate: "            << disaster_rate << endl;
        for (int i = 0; i < nz; i += 1) {
            outf << left << setw(sp_col-10) << "Distribution of entrants, z = "
                 << left << setw(10) << i  << distribution_of_entrants(i) << endl;
        }

        // Restore precision
        outf.precision(ss);
    };
};

//// Operator overloading
ostream & operator << (ostream &out, const StructuralParameters &c)
{
    cout << left << "STRUCTURAL PARAMETERS:" << endl << endl;
    cout << left << setw(sp_col) << "NAME" << "VALUE" << endl;
    cout << left << setw(sp_col) << "Number of productivity shocks: "      << c.nz << endl;
    cout << left << setw(sp_col) << "Maximum firm size: "                  << c.nbar << endl;
    cout << left << setw(sp_col) << "Low Wage: "                           << c.wl << endl;
    cout << left << setw(sp_col) << "High Wage: "                          << c.wh << endl;
    cout << left << setw(sp_col) << "Decreasing returns parameter: "       << c.decreasing_returns << endl;
    cout << left << setw(sp_col) << "Real interest rate: "                 << c.real_rate << endl;
    cout << left << setw(sp_col) << "Instantaneous fixed cost: "           << c.fixed_cost << endl;
    cout << left << setw(sp_col) << "Exit rate: "                          << c.exit_rate << endl;
    cout << left << setw(sp_col) << "Layoff rate: "                        << c.layoff_rate << endl;
    cout << left << setw(sp_col) << "Instantaneous severance pay: "        << c.layoff_cost << endl;
    cout << left << setw(sp_col) << "Exogenous separation rate: "          << c.exogenous_separations_rate << endl;
    cout << left << setw(sp_col) << "Aggregate productivity shifter: "     << c.aggregate_productivity << endl;
    cout << left << setw(sp_col) << "AR(1) rho: "                          << c.rho << endl;
    cout << left << setw(sp_col) << "AR(1) sigma_epsilon: "                << c.sigma << endl;
    cout << left << setw(sp_col) << "AR(1) mu: "                           << c.mu << endl;
    cout << left << setw(sp_col) << "Financial Disaster Rate: "            << c.disaster_rate << endl;

    cout << "\n\nENTRANTS DISTRIBUTION\n";
    cout << left << setw(20) <<"Productivity" << "Probability\n";
    VectorXd aux_p = c.distribution_of_entrants;
    VectorXd aux_z = c.productivity;
    for (int i = 0; i < c.nz; i += 1){
        cout << left << setw(20) << aux_z(i) << aux_p(i) <<"\n";
    }

    cout << "\n\nINFINITESIMAL GENERATOR OF POISSON PRODUCTIVITY PROCESS\n";
    cout << c.Q << "\n";

    return out;
}

istream & operator >> (istream &in,  StructuralParameters &c)
{
    // Ask the user for the parameters in sequential order (with defaults)
    string accept_defaults;
    bool answered = false;

    while (!answered) {

        cout << "Do you want to accept all of the default parameters? [y/n]\n";
        cin >> accept_defaults;
        if (accept_defaults == "y") {
            answered = true;
            // Set member values
            c.nz = default_nz;
            c.nbar = default_nbar;
            c.wl = default_wl;
            c.wh = default_wh;
            c.decreasing_returns = default_decreasing_returns;
            c.real_rate = default_real_rate;
            c.fixed_cost = default_fixed_cost;
            c.exit_rate = default_exit_rate;
            c.layoff_rate = default_layoff_rate;
            c.layoff_cost = default_layoff_cost;
            c.exogenous_separations_rate = default_exogenous_separations_rate;
            c.aggregate_productivity = default_aggregate_productivity;
            c.rho = default_rho;
            c.sigma = default_sigma;
            c.mu = default_mu;
            c.disaster_rate = default_disaster_rate;
            c.distribution_of_entrants = default_distribution_of_entrants;


        } else if (accept_defaults == "n") {
            answered = true;
            cout << "Excellent! I will ask you what numbers you want. Let's get started :) \n";
            cout << "[You can press [enter] to accept the default value!]\n\n";
            cout << "How many productivity shocks? [int, default = 4]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.nz;
            }

            cout << "What is the largest firm size? [int, default = 60]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.nbar;
            }
            cout << "Low wage? [double, default = 1]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.wl;
            }
            cout << "High wage? [double, default = 1.25]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.wh;
            }
            cout << "How concave should the production function be? [double, default = 0.66]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.decreasing_returns;
            }
            cout << "What is the real interest rate? [double, default = 0.01]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.real_rate;
            }
            cout << "How about the instantaneous fixed cost? [double, default = 1.5]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.fixed_cost;
            }
            cout << "What is the firm exit rate? [double, default = 100000]\n";
            cout << "[Remember, a rate of 1 corresponds to a mean arrival time of 1 quarter.]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.exit_rate;
            }
            cout << "What is the layoff rate? [double, default = 13]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.layoff_rate;
            }
            cout << "How many quarters of severance pay are there? [double, default = 1]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.layoff_cost;
            }
            cout << "What is the \033[1;37mexogenous\033[0m separation rate? [double, default = 1/16]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.exogenous_separations_rate;
            }
            cout << "In percentage terms, how higher/lower should aggregate productivity be? [double, default = 0, 1 means 1%]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                double agg;
                cin >> agg;
                c.aggregate_productivity = 1 + agg / 100;
            }
            cout << "What is the persistence of the AR process for productivity? [double, default = 0.6]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.rho;
            }
            cout << "What is the variance of the AR process for productivity? [double, default = 0.1]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.sigma;
            }
            cout << "What is the offset of the AR process for productivity? [double, default = 0.8]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.mu;
            }
            cout << "Tell me the rate at which firms become permanently insolvent. [double, default = 0.1]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.disaster_rate;
            }

            // Distribution of entrants.

            cout << "\n\nWe're almost there :). We need to take care of the distribution of entrants.\n";

            cout << "I see that you have chosen ";
            if (c.nz < 5){
                cout << c.nz << " productivity states. That is not too large a number.\n\n";
                cout << "Should I do this myself? [y/n]\n";
                answered = false;

                while (!answered) {
                    string populate_entrants;
                    cin >> populate_entrants;
                    if (populate_entrants == "y") {
                        answered = true;
                        cout << "I will assign a uniform distribution over the non-disaster states.\n";
                        VectorXd aux; aux = VectorXd::Constant(c.nz-1,1.0/(c.nz-1));
                        VectorXd distribution_of_entrants; distribution_of_entrants = VectorXd::Constant(c.nz,0);
                        for (int i = 1; i < c.nz ; i += 1){
                            distribution_of_entrants(i) = aux(i-1);
                        }
                        c.distribution_of_entrants = distribution_of_entrants;

                    } else if (populate_entrants == "n") {
                        answered = true;
                        VectorXd aux; aux = VectorXd::Constant(c.nz,0);
                        vector<string> position = {"first", "second", "third", "fourth"};
                        for (int i = 0; i < c.nz; i += 1){
                            double share;
                            cout << "Enter the fraction of firms who enter at the " << position[i] << " productivity level:\n";
                            cin >> share;
                            aux(i) = share;
                        }
                        c.distribution_of_entrants = aux;

                    } else {
                        cout << "Please enter [y] or [n]\n";
                    }
                }

            } else {
                cout << c.nz << " productivity states. That is too large a number for me, sorry.\n";
                cout << "I will assign a uniform distribution over the non-disaster states.\n";

                VectorXd aux; aux = VectorXd::Constant(c.nz-1,1.0/(c.nz-1));
                VectorXd distribution_of_entrants; distribution_of_entrants = VectorXd::Constant(c.nz,0);
                for (int i = 1; i < c.nz ; i += 1){
                    distribution_of_entrants(i) = aux(i-1);
                }
                c.distribution_of_entrants = distribution_of_entrants;
            }


//            cout << "\n\033[1;37mThank you so much! Have a nice day and see you soon :)\033[0m\n\n";

        } else {
            cout << "Please enter [y] or [n]\n";
        }
    }

    c.discount_rate = log(1.0 + c.real_rate);

//         Create Infinitesimal Generator of Poisson Process for Productivity
    DenseCTMC ig; ig.discretizeAR1_Rouwenhorst(c.rho, c.sigma, c.mu, c.nz - 1);
    MatrixXd TM = ig.getTransitionMatrix();
    for (int i = 0; i < c.nz - 1; i += 1){
        double factor = 1.0/TM(i,i);
        TM(i,i) += - c.disaster_rate;
        TM.row(i) = TM.row(i) * factor * TM(i,i);    // I want the total out rate to be just like before.
    }
    MatrixXd newTM(c.nz,c.nz);
    newTM.bottomRightCorner(c.nz-1,c.nz-1) = TM;
    newTM.row(0) = VectorXd::Constant(c.nz, 0);
    newTM.bottomLeftCorner(c.nz-1, 1) = VectorXd::Constant(c.nz-1, c.disaster_rate);
    c.Q = newTM;

    VectorXd aux = ig.getStateSpace();
    VectorXd Z; Z = VectorXd::Constant(c.nz,0); Z.tail(c.nz-1) = aux;
    for (int i = 1; i < c.nz; i += 1){
        Z(i) = aux(i-1);
    }
    c.productivity = Z;

    return in;
}




#endif
