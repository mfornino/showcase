//============================================================================
// Name        : main.cpp
// Description : Test temporary pieces of code or classes in the making.
// Date        : May 11, 2018
//============================================================================

#include <iostream>
#include <iomanip>
#include "src/StructuralParameters.h"
#include "src/TechnicalParameters.h"
#include "src/Model.h"

using namespace std;

int main () {


    // THIS SNIPPET SHOULD GO AT THE VERY BEGINNING OF THE MODEL!!!

    cout << "\n\033[1mWelcome to the initialization of the model parameters!\033[0m";

    StructuralParameters sParams;
    TechnicalParameters tParams;

    cout << "\nThe first step is to initialize the model's structural parameters.\n\n";
    cout << "Can I load them from a file?\nType the name of the file together with the extension, or press [enter] to continue.\n";
    string filename;
    if (cin.peek() != '\n') {
        cin >> filename;
        sParams = StructuralParameters(filename);
    } else {
        cin >> sParams;
    }

    cout << "\n\nThe second step is to initialize the technical parameters for the numerical exercise.\n\n";
    cout << "Can I load them from a file?\nType the name of the file together with the extension, or press [enter] to continue.\n";
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(),'\n');
    if (cin.peek() != '\n') {
        cin >> filename;
        tParams = TechnicalParameters(filename);
    } else {
        cin >> sParams;
    }




    Model model = Model(sParams, tParams, false);

    cout << "\n" << model.getStructuralParameters();
    cout << "\n" << model.getTechnicalParameters();


//    sParams.saveToFile();
//    tParams.saveToFile();

    return 0;
}
