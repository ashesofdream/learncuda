#include <random>
#include <iostream>
#include <cstdlib>
using namespace std;
#include "util.h"
std::random_device rd;
std::mt19937 gen(rd());

void util::init_array_int(int *ip , int size){
    std::uniform_int_distribution distrib;
    for(int i = 0; i < size;++i){
        ip[i] = distrib(gen)&0xFF;
    }
}

void util::print_matrix(int *array, int m, int n){
    int cnt = 0;
    for(int i = 0 ; i < m; ++i ){
        for(int j = 0 ; j < n ; ++j ){
            cout << array[cnt++] << " ";
        }
        cout<<endl;
    }
    cout<<endl;
}

