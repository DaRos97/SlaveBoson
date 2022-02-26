#ifndef __CLASS_H_
#define __CLASS_H_

#include <iostream>
#include <cmath>

using namespace std;

class myclass{
    int private_int;
    public:
    int public_int;

    //constructor
    myclass(int);
    myclass(int,int);

    //a function
    int class_func(int);

};


#endif
