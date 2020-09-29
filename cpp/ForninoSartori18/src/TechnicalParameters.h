//
// Created by Michele Fornino on 5/14/18.
//

#ifndef TECHNICALPARAMETERS_H
#define TECHNICALPARAMETERS_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include "markovchains/MarkovChain.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

using namespace std;
using namespace MarkovChain;

const string TECHNICALPARAMETERS_RELATIVE_PATH = "../user/";


// Parameters of parameters' file.
const int tp_col = 40;         // Set size of column in parameters file
const int tp_start = 3;        // start from line 4
const int tp_max_lines = 20;  // read through line 100

// Defaults defined as global for convenience
const double default_hjb_criterion = 1e-10;
const int default_hjb_maxit = 100;
const double default_hjb_step_size = 1e8;
const double default_hjb_criterion_BiCGSTAB = 1e-6;
const double default_kfe_criterion = 1e-10;
const double default_kfe_step_size = 1e8;


class TechnicalParameters
{

public:

    //// Member Parameters
    // Parameters for HJB
    double hjb_criterion;
    int hjb_maxit;
    double hjb_step_size;
    double hjb_criterion_BiCGSTAB; // Must be relative to criterion_hjb

    // Parameters for KFE
    double kfe_criterion;
    double kfe_step_size;

    //// Constructor to initialize reading appropriately defined file
    TechnicalParameters() {
        setDefaults();
    }

    TechnicalParameters(string &filename)
    {
        loadFromFile(filename);
    };

    // setDefaults Constructor helper
    void setDefaults() {
        // Parameters for HJB
        hjb_criterion = default_hjb_criterion;
        hjb_maxit = default_hjb_maxit;
        hjb_step_size = default_hjb_step_size;
        hjb_criterion_BiCGSTAB = default_hjb_criterion_BiCGSTAB; // Must be relative to criterion_hjb

        // Parameters for KFE
        kfe_criterion = default_kfe_criterion;
        kfe_step_size = default_kfe_step_size;
    }

    // loadFromFile Constructor Helper
    void loadFromFile(string &filename) {
        // Declare File Name
        string pathToFile = TECHNICALPARAMETERS_RELATIVE_PATH + filename;
        ifstream inf(pathToFile, ios::binary);

        if (!inf) {

            cout << "Could not load from file. Setting defaults.";
            setDefaults();

        } else {

            // Parse file. First grab
            string str;
            vector<double> inputs;
            int line = 0;
            while (line < tp_max_lines) {
                getline(inf, str);
                if (line >= tp_start && !str.empty()) {
                    inputs.emplace_back(atof(str.substr(40).c_str()));
                };
                line += 1;
            }

            // Set member values
            // Parameters for HJB
            hjb_criterion = inputs[0];
            hjb_maxit = (int)round(inputs[1]);
            hjb_step_size = inputs[2];
            hjb_criterion_BiCGSTAB = inputs[3];

            // Parameters for KFE
            kfe_criterion = inputs[4];
            kfe_step_size = inputs[5];
        }
    };

    //// Overloading stream insertion operators
    friend ostream & operator << (ostream &out, const TechnicalParameters &c);
    friend istream & operator >> (istream &in,  TechnicalParameters &c);

    //// Methods
    void saveToFile(string filename = "technicalParameters.txt") {

        // Declare File Name
        string pathToFile = TECHNICALPARAMETERS_RELATIVE_PATH + filename;
        system("mkdir -p ../user/");
        ofstream outf(pathToFile, ios::binary);

        // Obtain default precision of outf
        streamsize ss = outf.precision();

        // Set higher precision
        outf.precision(16);

        // Save Values
        outf << left << "TECHNICAL PARAMETERS:" << endl << endl;
        outf << left << setw(tp_col) << "NAME" << "VALUE" << endl;
        outf << left << setw(tp_col) << "HJB Equation Tolerance Criterion: "   << hjb_criterion << endl;
        outf << left << setw(tp_col) << "HJB Equation Maximum Iterations: "    << hjb_maxit << endl;
        outf << left << setw(tp_col) << "HJB Equation Implicit Step Size: "    << hjb_step_size << endl;
        outf << left << setw(tp_col) << "HJB Equation BiCGSTAB Tolerance: "    << hjb_criterion_BiCGSTAB << endl;
        outf << left << setw(tp_col) << "KFE Equations Tolerance Criterion: "  << kfe_criterion << endl;
        outf << left << setw(tp_col) << "KFE Equations Implicit Step Size: "   << kfe_step_size << endl;

        // Restore precision
        outf.precision(ss);
    };
};

//// Operator overloading
ostream & operator << (ostream &out, const TechnicalParameters &c)
{
    cout << left << "TECHNICAL PARAMETERS:" << endl << endl;
    cout << left << setw(tp_col) << "NAME" << "VALUE" << endl;
    cout << left << setw(tp_col) << "HJB Equation Tolerance Criterion: "   << c.hjb_criterion << endl;
    cout << left << setw(tp_col) << "HJB Equation Maximum Iterations: "    << c.hjb_maxit << endl;
    cout << left << setw(tp_col) << "HJB Equation Implicit Step Size: "    << c.hjb_step_size << endl;
    cout << left << setw(tp_col) << "HJB Equation BiCGSTAB Tolerance: "    << c.hjb_criterion_BiCGSTAB << endl;
    cout << left << setw(tp_col) << "KFE Equations Tolerance Criterion: "  << c.kfe_criterion << endl;
    cout << left << setw(tp_col) << "KFE Equations Implicit Step Size: "   << c.kfe_step_size << endl;

    return out;
}

istream & operator >> (istream &in,  TechnicalParameters &c)
{
    // Ask the user for the parameters in sequential order (with defaults)
    string accept_defaults;
    bool answered = false;

    while (!answered) {

        cout << "Do you want to accept all of the default parameters? [y/n]\n";
        cin >> accept_defaults;
        if (accept_defaults == "y") {
            answered = true;
            // Parameters for HJB
            c.hjb_criterion = default_hjb_criterion;
            c.hjb_maxit = default_hjb_maxit;
            c.hjb_step_size = default_hjb_step_size;
            c.hjb_criterion_BiCGSTAB = default_hjb_criterion_BiCGSTAB; // Must be relative to criterion_hjb

            // Parameters for KFE
            c.kfe_criterion = default_kfe_criterion;
            c.kfe_step_size = default_kfe_step_size;


        } else if (accept_defaults == "n") {
            answered = true;
            cout << "Excellent! I will ask you what numbers you want. Let's get started :) \n";
            cout << "[You can press [enter] to accept the default value!]\n\n";
            cout << "HJB Equation Tolerance Criterion. [double, default = 1e-10]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.hjb_criterion;
            }

            cout << "HJB Equation Maximum Iterations. [int, default = 100]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.hjb_maxit;
            }
            cout << "HJB Equation Implicit Step Size. [double, default = 1e8]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.hjb_step_size;
            }
            cout << "HJB Equation BiCGSTAB Tolerance. [double, default = 1e-6]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.hjb_criterion_BiCGSTAB;
            }
            cout << "How concave should the production function be? [double, default = 0.66]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.kfe_criterion;
            }
            cout << "What is the real interest rate? [double, default = 0.01]\n";
            cin.ignore();
            if (cin.peek() != '\n') {
                cin >> c.kfe_step_size;
            }

//            cout << "\n\033[1;37mThank you so much! Have a nice day and see you soon :)\033[0m\n\n";

        } else {
            cout << "Please enter [y] or [n]\n";
        }
    }
    return in;
}

#endif
