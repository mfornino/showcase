//
// Created by Michele Fornino on 5/10/18.
//


#include "StructuralParameters.h"
#include "TechnicalParameters.h"
#include "Eigen/Core"

#ifndef MODEL_H
#define MODEL_H


class Model {

private:

    StructuralParameters sp;
    TechnicalParameters tp;

    int num_tot_states;
    int num_endog_states;

    VectorXd value_guess;
//    VectorXd distribution_guess;


    MatrixX3i table_of_states;
    MatrixXi table_of_states_inv;

    // ADD PRIVATE VARIABLES THAT MUST BE PASSED BETWEEN METHODS


    // Create matrix for lookup between idx and labor force (not trivial mapping, will try to fix this in the future)





public:

    Model() = default;
    ~Model() = default;
    Model(StructuralParameters &c_sp, TechnicalParameters &c_tp);
    Model(StructuralParameters &c_sp, TechnicalParameters &c_tp, bool loadFromFile);

    void setDerived();
    void loadGuesses();


    StructuralParameters getStructuralParameters() { return sp; }
    TechnicalParameters getTechnicalParameters() { return tp; }





    // MEMBERS (variables and other useful objects)

    // ModelParameters m_model_parameters;
    // TechnicalParameters m_technical_parameters;
    //


    // STRUCT OBJECTS TO KEEP STUFF ORGANIZED

    // struct for structural parameters (initialize + default values)


    // struct for technical parameters


    // struct for endogenous variables
    struct EndogenousVariables
    {
        // doubles for
        // hL
        // hH
        // q
        // M
    };

    
    // struct for output of HJB equation
    struct OutputHJB
    {
        // value function
        // Infinitesimal generator (adjusted for entry and not adjusted)

    };

    struct SparseMatricesHJB
    {
        // structure should contain the (triplets for) the sparse matrices needed to perform HJB
    };


    struct OutputStatistics
    {
        // structure contains statistics
    };


    struct OutputStationaryEquations
    {
        // structure contains residuals of equations determining the stationary equilibrium
    };


    // CONSTRUCTORS
    // Initialize class using a bunch of default values/ModelParameters struct


    // Note: Endogenous Variables should be an input to the methods, so that the underlying instantiation of the
    // class Model does not change.

    // Model(StructuralParameters struct_params)
    // Model(TechnicalParameters tech_params)
    // Model(StructuralParameters struct_params, TechnicalParameters tech_params)




    // METHODS



    // Create sparse matrices for updated endogenous variables

    // Solve HJB
    // OutputHJB solveHJBwithGuess(VectorXd guess)

};


#endif
