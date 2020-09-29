//
// Created by Michele Fornino on 5/10/18.
//

#include "Model.h"

Model::Model(StructuralParameters &c_sp, TechnicalParameters &c_tp){
    sp = c_sp;
    tp = c_tp;
    setDerived();
}

Model::Model(StructuralParameters &c_sp, TechnicalParameters &c_tp, bool loadFromFile){
    sp = c_sp;
    tp = c_tp;
    setDerived();
    if (loadFromFile) loadGuesses();
}

void Model::setDerived(){
    // Number of endogenous and total states
    num_endog_states = (sp.nbar + 1) * (sp.nbar + 2) / 2;
    num_tot_states = sp.nz * num_endog_states;

    // Lookup matrices for states/labor force
    MatrixX3i c_table_of_states(num_endog_states, 3);
    MatrixXi c_table_of_states_inv = MatrixXi::Zero(sp.nbar+1, sp.nbar+1);
    int k = 0;
    for (int j = 0; j <= sp.nbar; j += 1) {
        for (int i = 0; i <= sp.nbar; i += 1) {
            if (i + j <= sp.nbar) {
                c_table_of_states(k,0) = k;
                c_table_of_states(k,1) = i;
                c_table_of_states(k,2) = j;
                c_table_of_states_inv(i,j) = k;
                k += 1;
            }
        }
    }
    table_of_states = c_table_of_states;
    table_of_states_inv = c_table_of_states_inv;


}


void Model::loadGuesses() {

    // TODO

}