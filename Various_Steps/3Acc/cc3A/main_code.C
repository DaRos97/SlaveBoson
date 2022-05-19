#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include "functions.C"
#include "class.C"
#include "inputs.C"

using namespace std;
//using std::vector;

int main(){
    //
    Double* energies = new Double[2*PD*PD];
    for(int ans=0;ans<2;ans++){
        for(int j2 = 0;j2<PD;j2++){
            for(int j3=0;j3<PD;j3++){
                // real code her
                Double params[3] = { rJ2[j2], rJ3[j3], ans};
                energies[ans*PD*PD+j2*PD+j3] = j2;
            }
        }
        cout<<energies[0]<<endl;
    }
    
    delete[] energies;
}
