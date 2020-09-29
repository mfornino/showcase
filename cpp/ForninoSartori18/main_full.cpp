#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "src/markovchains/MarkovChain.h"

using namespace Eigen;
using namespace std;
using namespace MarkovChain;

// Eigen Flags
//#define EIGEN_NDEBUG
//#define EIGEN_NO_STATIC_ASSERT


//// ENUMERATED CLASS FOR POLICY TYPE
enum class Policies
{
    IDLE,
    HIRE_LOW_WAGE,
    HIRE_HIGH_WAGE,
    LAYOFF_LOW_WAGE,
    LAYOFF_HIGH_WAGE,
    EXIT,
};

//// FUNCTION PHI THAT RETURNS SCALING OF HIRING ENTRY BY FIRM SIZE (PART OF THE PARAMETERS OF THE MODEL!!!)
double phi(int total_employment){
    return total_employment + 1;
}


//// MAIN PROGRAM
int main()
{
    const bool verbose = true;
    const bool graphs = true;
    const bool tables = false;
    const bool load_guess_from_file = false;
    const bool save_guess_to_file = false;

    // Greetings
    if (verbose) {
        cout <<"----------------------------------------------\n";
        cout <<"------| SOLVING THE PROBLEM OF THE FIRM |-----\n";
        cout <<"----------------------------------------------\n\n";
    }

    // Start Clock
    auto t0 = std::chrono::high_resolution_clock::now();

    // Endogenous Variables (should be passed as arguments of the function)
    double hiring_rate_low = 0.1;
    double hiring_rate_high = 3.0;
    double quit_rate_low = 0.07;

    // Parameters of the state space
    const int nw = 2;       // ASSUME N_w = 2 for the time being
    const int nz = 4;       // For now fix it at nz = 4
    const int nbar = 100;
    const double wl = 1.0;
    const double wh = 1.25;

    // Parameters of the model (read from file)
    const double decreasing_returns = 0.666666666;
    const double real_rate = 0.01;
    const double discount_rate = log(1.0 + real_rate);
    const double fixed_cost = 1.5;
    const double exit_rate = 1e5;
    const double layoff_rate = 13.0;
    const double layoff_cost = 1.0;
    const double exogenous_separations_rate = 1.0/16.0;
    const double aggregate_productivity = 1.0;

    // Parameters for HJB
    const double hjb_criterion = 1e-10;
    const int hjb_maxit = 100;
    const double hjb_step_size = 1e8;
    const double hjb_criterion_BiCGSTAB = 1e-6; // Must be relative to criterion_hjb

    // Parameters for KFE
    const double kfe_criterion = 1e-10;
//    const int kfe_maxit = 100;
    const double kfe_step_size = 1e8;
    const double criterion_kfe_BiCGSTAB = 1e-6; // Must be relative to criterion_hjb

    // Infinitesimal Generator of Poisson Process for Productivity and set \mathcal{Z}.
    Matrix4d Q;
    Q(0,0) = -0.00; Q(0,1) =  0.00; Q(0,2) =  0.00; Q(0,3) =  0.00;
    Q(1,0) =  0.10; Q(1,1) = -0.55; Q(1,2) =  0.30; Q(1,3) =  0.15;
    Q(2,0) =  0.10; Q(2,1) =  0.25; Q(2,2) = -0.60; Q(2,3) =  0.25;
    Q(3,0) =  0.10; Q(3,1) =  0.15; Q(3,2) =  0.30; Q(3,3) = -0.55;

    Vector4d productivity;
    productivity(0) = 0.00;
    productivity(1) = 2.00;
    productivity(2) = 2.75;
    productivity(3) = 3.50;

    // Entrants
    Vector4d distribution_of_entrants;
    distribution_of_entrants(0) = 0.00;
    distribution_of_entrants(1) = 0.25;
    distribution_of_entrants(2) = 0.50;
    distribution_of_entrants(3) = 0.25;

    // ADD PRINTING OF PARAMETERS
    if (verbose){
        // PRINT PARAMETERS HERE
    }

    // Type Definitions for Clarity
    typedef Triplet<double> T;
    typedef SparseMatrix<double, ColMajor> SM;
    typedef SparseMatrix<double, RowMajor> SMr;

    //// STEP 1: CREATE SPARSE MATRICES & PROFIT VECTORS
    // NOTE: formulation removes for the first time all states such that n_l(omega) + n_h(omega) > n_bar.

    if (verbose){
        cout <<"--------| CREATING SPARSE MATRICES... |-------\n";
    }

    const int num_endog_states = (nbar + 1) * (nbar + 2) / 2;
    const int num_tot_states = nz * num_endog_states;

    // Create matrix for lookup between idx and labor force (not trivial mapping, will try to fix this in the future)
    MatrixX3i table_of_states(num_endog_states, 3);
    MatrixXi table_of_states_inv = MatrixXi::Zero(nbar+1, nbar+1);
    int k = 0;
    for (int j = 0; j <= nbar; j += 1) {
        for (int i = 0; i <= nbar; i += 1) {
            if (i + j <= nbar) {
                table_of_states(k,0) = k;
                table_of_states(k,1) = i;
                table_of_states(k,2) = j;
                table_of_states_inv(i,j) = k;
                k += 1;
            }
        }
    }


    // Define triplets + sparse matrices, and vectors for profits and severance pay
    vector<T> triplet_stay; triplet_stay.reserve(num_tot_states * (2 * nz + 6));
    vector<T> triplet_layoff_low; triplet_layoff_low.reserve(2 * num_tot_states);
    vector<T> triplet_layoff_high; triplet_layoff_high.reserve(2 * num_tot_states);
    vector<T> triplet_hire_low; triplet_hire_low.reserve(2 * num_tot_states);
    vector<T> triplet_hire_high; triplet_hire_high.reserve(2 * num_tot_states);
    vector<T> triplet_exit; triplet_exit.reserve(num_tot_states);
    SM stay(num_tot_states,num_tot_states);
    SM layoff_low(num_tot_states,num_tot_states);
    SM layoff_high(num_tot_states,num_tot_states);
    SM hire_low(num_tot_states,num_tot_states);
    SM hire_high(num_tot_states,num_tot_states);
    SM exit(num_tot_states,num_tot_states);
    VectorXd profits(num_tot_states);
    VectorXd severance_pay_low(num_tot_states);
    VectorXd severance_pay_high(num_tot_states);

    // Populate objects by looping over all states, endogenous and exogenous.
    for (int i = 0; i < num_tot_states; i += 1) {
        int idx_endog = i % num_endog_states;
        int nl = table_of_states(idx_endog, 1);
        int nh = table_of_states(idx_endog, 2);
        int n = nl + nh;
        int z = i / num_endog_states;
        if ( nl != 0) {
            triplet_stay.emplace_back(T(i, i, - (exogenous_separations_rate + quit_rate_low) * nl));
            triplet_stay.emplace_back(T(i, i - 1, (exogenous_separations_rate + quit_rate_low) * nl));
            triplet_layoff_low.emplace_back(T(i, i, -layoff_rate * nl));
            triplet_layoff_low.emplace_back(T(i, i - 1, layoff_rate * nl));
        }
        if (nh != 0) {
            triplet_stay.emplace_back(T(i, i, - exogenous_separations_rate * nh));
            triplet_stay.emplace_back(T(i, i - (nbar - nh + 2), exogenous_separations_rate * nh));
            triplet_layoff_high.emplace_back(T(i, i  , -layoff_rate * nh));
            triplet_layoff_high.emplace_back(T(i, i - (nbar - nh + 2), layoff_rate * nh));
        }
        if (n != nbar) {
            triplet_hire_low.emplace_back(T(i, i, - hiring_rate_low * phi(n)));
            triplet_hire_low.emplace_back(T(i, i + 1, hiring_rate_low * phi(n)));
            triplet_hire_high.emplace_back(T(i, i, - hiring_rate_high * phi(n)));
            triplet_hire_high.emplace_back(T(i, i + (nbar - nh + 1), hiring_rate_high * phi(n)));
        }
        triplet_exit.emplace_back(T(i, i, - exit_rate));
        for (int k = 0; k < nz; k += 1) {
            triplet_stay.emplace_back(T(i, i + (k - z) * num_endog_states, Q(z,k)));
        }
        profits(i) = productivity(z) * pow(nl + nh, decreasing_returns) - wl * nl - wh * nh - fixed_cost;
        severance_pay_low(i) = -layoff_cost * layoff_rate * wl * nl;
        severance_pay_high(i) = -layoff_cost * layoff_rate * wh * nh;
    }
    stay.setFromTriplets(triplet_stay.begin(), triplet_stay.end());
    layoff_low.setFromTriplets(triplet_layoff_low.begin(), triplet_layoff_low.end());
    layoff_high.setFromTriplets(triplet_layoff_high.begin(), triplet_layoff_high.end());
    hire_low.setFromTriplets(triplet_hire_low.begin(), triplet_hire_low.end());
    hire_high.setFromTriplets(triplet_hire_high.begin(), triplet_hire_high.end());
    exit.setFromTriplets(triplet_exit.begin(), triplet_exit.end());

    // Time
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sparse = t1 - t0;
    if (verbose) {
        cout <<"Sparse Matrices Computed in: " << elapsed_sparse.count() << "\n\n";
        cout <<"--------| SOLVING THE HJB EQUATION... |-------\n";
    }

    //// STEP 2: SOLVE HJB EQUATION


    // Set guess of value function or try loading it from file.
    VectorXd value = VectorXd::LinSpaced(num_tot_states, 0, 1);
    if (load_guess_from_file) {
        ifstream inf("value_guess.dat", ios::binary);
        if (!inf) {
            ;
        } else {
            int nbar_file;
            int nz_file;
            inf >> nbar_file;
            inf >> nz_file;
            if (nbar_file == nbar && nz_file == nz){
                int i = 0;
                while (inf && i < num_tot_states) {
                    inf >> value(i);
                    i += 1;
                }
            }
        }
    }

    // Preallocate objects for HJB equation loop
    VectorXd value_next(num_tot_states);
    vector<Policies> policy;
    policy.reserve(num_tot_states);
    VectorXd profits_vector_for_hjb(num_tot_states);
    MatrixXi::Index maxRow, maxCol;
    Matrix<double, Dynamic, Dynamic> alternatives(num_tot_states, 6);

    VectorXd b(num_tot_states);

    // Identity Matrix
    SMr speye(num_tot_states,num_tot_states);
    speye.setIdentity();

    // Infinitesimal Generator (no entry)
    SMr sparse_matrix_for_hjb(num_tot_states,num_tot_states);
    sparse_matrix_for_hjb.reserve(VectorXi::Constant(num_tot_states, 4 + nz - 1));

    // Auxiliary Matrix for Linear System

    vector<T> triplet_sparse_matrix_for_hjb;
    triplet_sparse_matrix_for_hjb.reserve(num_tot_states * (nz * 2 + 6));

    // Solver Definition
    // BiCGSTAB
    BiCGSTAB<SMr> hjb_solver;
    hjb_solver.setTolerance(hjb_criterion_BiCGSTAB * hjb_criterion);

    // While Loop
    double max_diff = -1;
    double diff = -1;
    int iteration_hjb = 0;
    double supnorm_hjb = 1e100;
    while (iteration_hjb < hjb_maxit && supnorm_hjb > hjb_criterion) {

        // Construct Alternatives Matrix
        // Note: matrices hire_* layoff_* exit **do not** already contain stay components common to all choices.
        alternatives.col(0) = profits + stay * value;
        alternatives.col(1) = alternatives.col(0) + hire_low * value;
        alternatives.col(2) = alternatives.col(0) + hire_high * value;
        alternatives.col(3) = alternatives.col(0) + severance_pay_low + layoff_low * value;
        alternatives.col(4) = alternatives.col(0) + severance_pay_high + layoff_high * value;
        alternatives.col(5) = alternatives.col(0) + exit * value;

        // Solve out max operator and create transition matrix and vector
        profits_vector_for_hjb = profits;

        vector<T> triplet_update;
        triplet_update.reserve(num_tot_states * (2 * nz + 6));

        SMr A(num_tot_states,num_tot_states);
        A.reserve(num_tot_states * (nz + 4));
//        A.setZero();
        sparse_matrix_for_hjb.setZero();
        for (int i = 0; i < num_tot_states; i += 1) {
            double placeholder;
            int idx_endog = i % num_endog_states;
            int nl = table_of_states(idx_endog, 1);
            int nh = table_of_states(idx_endog, 2);
            int n = nl + nh;
            placeholder = alternatives.row(i).maxCoeff(&maxRow, &maxCol);
            policy[i] = static_cast<Policies>(maxCol);
            switch (policy[i])
            {
                case Policies::IDLE:
                    break;
                case Policies::HIRE_LOW_WAGE:
                    triplet_update.emplace_back(T(i, i, -hiring_rate_low * (n + 1)));
                    triplet_update.emplace_back(T(i, i + 1, hiring_rate_low * (n + 1)));
                    break;
                case Policies::HIRE_HIGH_WAGE:
                    triplet_update.emplace_back(T(i, i, -hiring_rate_high * (n + 1)));
                    triplet_update.emplace_back(T(i, i + (nbar - nh + 1), hiring_rate_high * (n + 1)));
                    break;
                case Policies::LAYOFF_LOW_WAGE:
                    triplet_update.emplace_back(T(i, i, -layoff_rate * nl));
                    triplet_update.emplace_back(T(i, i - 1, layoff_rate * nl));
                    profits_vector_for_hjb(i) += severance_pay_low(i);
                    break;
                case Policies::LAYOFF_HIGH_WAGE:
                    triplet_update.emplace_back(T(i, i  , -layoff_rate * nh));
                    triplet_update.emplace_back(T(i, i - (nbar - nh + 2), layoff_rate * nh));
                    profits_vector_for_hjb(i) += severance_pay_high(i);
                    break;
                case Policies::EXIT:
                    triplet_update.emplace_back(T(i, i  , -exit_rate));
                    break;
            }
        }
        triplet_update.insert(triplet_update.end(), triplet_stay.begin(), triplet_stay.end());
        sparse_matrix_for_hjb.setFromTriplets(triplet_update.begin(), triplet_update.end());

        // Solve Linear System
        b = profits_vector_for_hjb + (1 / hjb_step_size) * value;
        A = (1 / hjb_step_size + discount_rate) * speye - sparse_matrix_for_hjb;

        // BiCGSTAB
        hjb_solver.compute(A);
        value_next = hjb_solver.solveWithGuess(b,value);

        // Compute Supnorm
        max_diff = -1;
        diff = -1;
        for (int i = 0; i < num_tot_states; i += 1){
            diff = abs(value(i) - value_next(i));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        supnorm_hjb = max_diff;

        // Update value function
        value = value_next;

        // Update iteration count and display advancement
        iteration_hjb += 1;
        if (verbose) {
            if (iteration_hjb % 5 == 0 || iteration_hjb ==1){
                cout <<"Iteration = "<<iteration_hjb<<", SupNorm = "<<supnorm_hjb<<"\n";
            }
        }

        // Save vector of triplets for fast creation of final sparse matrix
        if (supnorm_hjb < hjb_criterion) {
            triplet_sparse_matrix_for_hjb.insert(triplet_sparse_matrix_for_hjb.end(), triplet_update.begin(), triplet_update.end());
        }
    }

    // Add entrants to the Sparse Infinitesimal Generator
    for (int i = 0; i < num_tot_states; i += 1) {
        switch (policy[i])
        {
            case Policies::IDLE:break;
            case Policies::HIRE_LOW_WAGE:break;
            case Policies::HIRE_HIGH_WAGE:break;
            case Policies::LAYOFF_LOW_WAGE:break;
            case Policies::LAYOFF_HIGH_WAGE:break;
            case Policies::EXIT:
                for (int k = 0; k < nz; k += 1) {
                    triplet_sparse_matrix_for_hjb.emplace_back(T(i, num_endog_states * k, exit_rate * distribution_of_entrants(k)));
                }
                break;
        }
    }
    sparse_matrix_for_hjb.setZero();
    sparse_matrix_for_hjb.setFromTriplets(triplet_sparse_matrix_for_hjb.begin(), triplet_sparse_matrix_for_hjb.end());


    // Save Value Function to File
    if (save_guess_to_file) {
        ofstream outf("value_guess.dat", ios::binary);
        if (!outf) {
            ;
        } else {
            std::streamsize ss = std::cout.precision();
            outf.precision(16);
            outf << nbar << endl;
            outf << nz << endl;
            for(int i =0; i < num_tot_states; i += 1) {
                outf << value(i) << endl;
            }
            outf.precision(ss);
        }
    }

    // Time
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_hjb = t2 - t1;
    if (verbose) {
        if (iteration_hjb % 5 != 0) {
            cout << "Iteration = " << iteration_hjb << ", SupNorm = " << supnorm_hjb << "\n";
        }
        cout <<"----------------------------------------------\n";
        cout <<"HJB Equations solved in: " << elapsed_hjb.count() << "\n\n";
        cout <<"--------| SOLVING THE KF EQUATIONS... |-------\n";
    }


    //// STEP 3: SOLVE KF EQUATIONS

    // Solve KF Equations using Sparse CTMC Class (REMEMBER!! A is the infinitesimal generator, SparseCTMC takes care
    // of getting the adjoint).
    SparseCTMC endogenous_ctmc = SparseCTMC(sparse_matrix_for_hjb);
    VectorXd distribution;
    distribution = endogenous_ctmc.getStationaryDistribution(kfe_step_size);
    if (verbose) cout << "Check Sum to 1:" << distribution.sum() << "\n";

    // Polish distribution trimming low values
    double cumsum = 0.0;
    for (int i = 0; i < num_tot_states; i += 1){
        if (abs(distribution(i)) < kfe_criterion) {
            distribution(i) = 0.0;
        }
        cumsum += distribution(i);
    }
    distribution = distribution / cumsum;


    // Time
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_kfe = t3 - t2;
    if (verbose) {
        cout <<"KF Equations solved in: " << elapsed_kfe.count() << "\n\n";
        cout <<"--------|   COMPUTING AGGREGATES...   |-------\n";
    }

    //// STEP 4: COMPUTE AGGREGATES
    //TODO: Add aggregates
    // Aggregates: Demand Intensities and Employment Shares
    double agg_hiring_flow_low_wage = 0;
    double agg_hiring_flow_high_wage = 0;
    double agg_demand_intensity_low_wage = 0;
    double agg_demand_intensity_high_wage = 0;
    double agg_employment_low_wage = 0;
    double agg_employment_high_wage = 0;

    // Statistics by Firm Size
    VectorXd firm_size_distribution = VectorXd::Zero(nbar + 1);
    VectorXd mean_wage_by_firm_size_temp = VectorXd::Zero(nbar + 1);
    VectorXd layoffs_low_wage_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd layoffs_high_wage_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd layoffs_total_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd mass_layoffs_low_wage_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd mass_layoffs_high_wage_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd mass_layoffs_total_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd endogenous_separations_by_firm_size = VectorXd::Zero(nbar + 1);
    VectorXd poaching_by_firm_size = VectorXd::Zero(nbar + 1);

    // Poaching
    VectorXd poaching_by_state(num_tot_states);
    double agg_fraction_poached;
    double agg_flow_of_quits;
    // TODO ADD POACHING


    // NL - NH graph unconditional on productivity level
    VectorXd unconditional_distribution = VectorXd::Zero(num_endog_states);

    for (int i = 0; i < num_tot_states; i += 1){
        int idx_endog = i % num_endog_states;
        int nl = table_of_states(idx_endog, 1);
        int nh = table_of_states(idx_endog, 2);
        int n = nl + nh;
        int z = i / num_endog_states;

        agg_employment_low_wage += nl * distribution(i);
        agg_employment_high_wage += nh * distribution(i);

        unconditional_distribution(idx_endog) += distribution(i);

        firm_size_distribution(n) += distribution(i);

        mean_wage_by_firm_size_temp(n) += (wl * nl + wh * nh) * distribution(i) / n;

        endogenous_separations_by_firm_size(n) += quit_rate_low * nl * distribution(i);


        switch (policy[i]){
            case Policies::HIRE_HIGH_WAGE:
                agg_demand_intensity_high_wage += distribution(i) * phi(n) * hiring_rate_high;
                break;
            case Policies::HIRE_LOW_WAGE:
                agg_demand_intensity_low_wage += distribution(i) * phi(n) * hiring_rate_low;
                break;
            case Policies::IDLE:break;
            case Policies::LAYOFF_LOW_WAGE:
                layoffs_low_wage_by_firm_size(n) += distribution(i) * nl * layoff_rate;
                layoffs_total_by_firm_size(n) += distribution(i) * nl * layoff_rate;
                break;
            case Policies::LAYOFF_HIGH_WAGE:
                layoffs_high_wage_by_firm_size(n) += distribution(i) * nh * layoff_rate;
                layoffs_total_by_firm_size(n) += distribution(i) * nh * layoff_rate;
                break;
            case Policies::EXIT:
                mass_layoffs_low_wage_by_firm_size(n) += distribution(i) * nl * exit_rate;
                mass_layoffs_high_wage_by_firm_size(n) += distribution(i) * nh * exit_rate;
                break;
        }



    }
    mass_layoffs_total_by_firm_size = mass_layoffs_low_wage_by_firm_size + mass_layoffs_high_wage_by_firm_size;
    agg_demand_intensity_low_wage = agg_hiring_flow_high_wage / hiring_rate_high;
    agg_demand_intensity_low_wage = agg_hiring_flow_low_wage / hiring_rate_low;


    int max_firm_size = 0;
    for (int i = 0; i <= nbar; i += 1){
        if (firm_size_distribution(i) > kfe_criterion) {
            max_firm_size = i;
        }
    }
    VectorXd mean_wage_by_firm_size(max_firm_size + 1);
    mean_wage_by_firm_size = mean_wage_by_firm_size_temp.head(max_firm_size + 1);
    for (int i = 0; i <= max_firm_size; i += 1){
        mean_wage_by_firm_size(i) = mean_wage_by_firm_size(i) / firm_size_distribution(i);
    }

    // Time
    auto t4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_agg = t4 - t3;
    if (verbose) {
        cout <<"Aggregates computed in: " << elapsed_agg.count() << "\n\n";
    }

    //// TOTAL TIME ELAPSED
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tot = t_end - t0;
    if (verbose) {
        cout <<"--------|            DONE!            |-------\n";
        cout << "Total Execution Time = " << elapsed_tot.count() << endl;
        cout <<" \n\n";
    }


    //// DISPLAY RESULTS FOR CLARITY

    //// TABLES
    if (tables) {
        cout << "----------------------------------------------\n";
        cout << "--------|    STATISTICS OF INTEREST   |-------\n";
        cout << "----------------------------------------------\n\n";

        cout << "Aggregate Employment:\n";
        cout << "Low Wage\t" << agg_employment_low_wage << "\n";
        cout << "High Wage\t" << agg_employment_high_wage << "\n";



        cout << "\n\n";
    }

    //// GRAPHS
    if (graphs) {
        cout << "----------------------------------------------\n";
        cout << "--------|            PLOTS            |-------\n";
        cout << "----------------------------------------------\n\n";

        //// POLICY FUNCTIONS
        cout << "--------|       POLICY FUNCTIONS      |-------\n\n";
        cout << "\033[1mLegend\033[0m\n";
        cout << "Color\t\tCode\tPolicy\n";
        cout << "\033[1;34mBlue\t\t2\t\tHire at High Wage\033[0m\n";
        cout << "\033[1;36mCyan\t\t1\t\tHire at Low Wage\033[0m\n";
        cout << "\033[1;32mGreen\t\t0\t\tStay\033[0m\n";
        cout << "\033[1;35mMagenta\t\t3\t\tLay Off at Low Wage\033[0m\n";
        cout << "\033[1;31mRed\t\t\t4\t\tLay Off at High Wage\033[0m\n";
        cout << "\033[1;30mWhite\t\t5\t\tExit\033[0m\n";

        for (int z = 0; z < nz; z += 1){
            cout << "\n\n";
            cout << "Productivity is: " << productivity(z) << "\n";
            for (int row = nbar; row >= 0; row -= 1){
                if (row == nbar) {
                    cout << "NH\t" << row << "\t";
                } else {
                    cout << "\n\t" << row <<"\t";
                }
                for (int col = 0; col <= nbar; col += 1){
                    if (row + col <= nbar) {
                        int index_endog = table_of_states_inv(col,row);
                        int i = z * num_endog_states + index_endog;
                        switch (policy[i]) {
                            case Policies::IDLE:
                                cout << "\033[1;32m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                            case Policies::HIRE_LOW_WAGE:
                                cout << "\033[1;36m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                            case Policies::HIRE_HIGH_WAGE:
                                cout << "\033[1;34m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                            case Policies::LAYOFF_LOW_WAGE:
                                cout << "\033[1;35m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                            case Policies::LAYOFF_HIGH_WAGE:
                                cout << "\033[1;31m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                            case Policies::EXIT:
                                cout << "\033[1;30m" << static_cast<int>(policy[i]) << "\033[0m\t"; break;
                        }
                    } else if (col < nbar){
                        cout << "\t";
                    }
                }
            }
            cout << "\n\t\t";
            for (int col = 0; col <= nbar; col += 1){
                cout << col << "\t";
            }
            cout << "\n\t\tNL\n";
        }

        //// UNCONDITIONAL DISTRIBUTION
        cout << "\n\n";

        cout << "--------| UNCONDITIONAL DISTRIBUTION  |-------\n\n";
        double graph_max = unconditional_distribution.maxCoeff();
        VectorXd graph_cutoffs = VectorXd::LinSpaced(7,kfe_criterion,graph_max+1e-6);
        string colors[7] = {"35", "31", "33", "32", "36", "34", "30"};

        cout << "\033[1mLegend\033[0m\n";
        cout << "Color\t\tInterval\n";
        cout << "\033[1;34mBlue\t\t[ " << graph_cutoffs(5) << ", " << graph_cutoffs(6) << " ]\033[0m\n";
        cout << "\033[1;36mCyan\t\t[ " << graph_cutoffs(4) << ", " << graph_cutoffs(5) << " )\033[0m\n";
        cout << "\033[1;32mGreen\t\t[ " << graph_cutoffs(3) << ", " << graph_cutoffs(4) << " )\033[0m\n";
        cout << "\033[1;33mGreen\t\t[ " << graph_cutoffs(2) << ", " << graph_cutoffs(3) << " )\033[0m\n";
        cout << "\033[1;31mRed\t\t\t[ " << graph_cutoffs(1) << ", " << graph_cutoffs(2) << " )\033[0m\n";
        cout << "\033[1;35mMagenta\t\t( " << "0" << ", " << graph_cutoffs(1) << " )\033[0m\n";
        cout << "\033[1;30mWhite\t\tLess than tolerance (essentially zero).\033[0m\n\n";

        for (int row = nbar; row >= 0; row -= 1){
            if (row == nbar) {
                cout << "NH\t" << row << "\t";
            } else {
                cout << "\n\t" << row << "\t";
            }
            for (int col = 0; col <= nbar; col += 1){
                if (row + col <= nbar) {
                    int index_endog = table_of_states_inv(col,row);
                    for (int j = 1; j < 7; j += 1){
                        if (unconditional_distribution(index_endog) >= kfe_criterion && unconditional_distribution(index_endog) < graph_cutoffs(j)){
                            cout << "\033[1;"<< colors[j-1] <<"mx\033[0m\t";
                            break;
                        } else if (unconditional_distribution(index_endog) < kfe_criterion){
                            cout << "\033[1;"<< colors[6] <<"mx\033[0m\t";
                            break;
                        }
                    }
                } else if (col < nbar){
                    cout << "\t";
                }
            }
        }
        cout << "\n\t\t";
        for (int col = 0; col <= nbar; col += 1){
            cout << col << "\t";
        }
        cout << "\n\t\tNL\n";



        //// FIRM SIZE DISTRIBUTION
        cout << "\n\n";
        cout << "--------|    FIRM SIZE DISTRIBUTION   |-------\n\n";

        graph_max = firm_size_distribution.maxCoeff();
        graph_cutoffs = VectorXd::LinSpaced(21,kfe_criterion,graph_max);
        for (int row = 21; row > 0; row -= 1){
            std::streamsize ss = std::cout.precision();
            if (row == 21) {
                cout.precision(2);
                cout << "f(x)" << std::fixed << graph_cutoffs(20) << "\t";
                cout.precision(ss);
            } else {
                cout.precision(2);
                cout << "\n\t" << std::fixed << graph_cutoffs(row-1) <<"\t";
                cout.precision(ss);
            }
            for (int col = 0; col <= nbar; col += 1){
                if (firm_size_distribution(col) < graph_cutoffs(row-1)){
                    cout << "\t";
                } else {
                    cout << "\033[1;34mx\033[0m\t";
                }
            }
        }
        cout << "\n\t\t\t";
        for (int col = 0; col <= nbar; col += 1){
            cout << col << "\t";
        }
        cout << "\n\t\t\tTotal Employees\n";



        //// MEAN WAGE BY FIRM SIZE
        cout << "\n\n";
        cout << "--------|    MEAN WAGE BY FIRM SIZE   |-------\n\n";

        graph_cutoffs = VectorXd::LinSpaced(21,wl,wh);
        for (int row = 21; row > 1; row -= 1){
            std::streamsize ss = std::cout.precision();
            if (row == 21) {
                cout.precision(2);
                cout << "f(x)" << std::fixed << graph_cutoffs(20) << "\t";
                cout.precision(ss);
            } else {
                cout.precision(2);
                cout << "\n\t" << std::fixed << graph_cutoffs(row-1) <<"\t";
                cout.precision(ss);
            }
            for (int col = 1; col <= max_firm_size; col += 1){
                if (mean_wage_by_firm_size(col) > graph_cutoffs(row-1) || mean_wage_by_firm_size(col) <= graph_cutoffs(row-2)){
                    cout << "\t";
                } else {
                    cout << "\033[1;34mx\033[0m\t";
                }
            }
        }
        cout << "\n\t\t\t";
        for (int col = 1; col <= max_firm_size; col += 1){
            cout << col << "\t";
        }
        cout << "\n\t\t\tTotal Employees\n";


        //// TODO ADD OTHER GRAPHS


    }
}