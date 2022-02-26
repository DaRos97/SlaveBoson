using Double = double;       //later can change it to long double
using Int = int;


double S = 0.5;
int kp = 20;
int PD = 5;


Double J2i = -0.3;
Double J2f = 0.3;
Double J3i = -0.3;
Double J3f = 0.3;

Double* rJ2 = new Double[PD];
Double* rJ3 = rJ2;

for(int i=0;i<PD;i++){
    //rJ2[i] = (J2f-J2i)/PD*i + J2i;
    //rJ3[i] = (J3f-J3i)/PD*i + J3i;
}

