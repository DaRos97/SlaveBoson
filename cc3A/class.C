#include "header.h"

using namespace std;

myclass::myclass(int p): private_int(p){
    public_int = 2;
}

myclass::myclass(int p, int P): private_int(p),public_int(P){}

int myclass::class_func(int k){
    int res;
    res = k + private_int;
    return res;
}

