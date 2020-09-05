// https://stackoverflow.com/questions/610245/where-and-why-do-i-have-to-put-the-template-and-typename-keywords

// =======================Name Lookup====================

// How does the compiler find  "t::x" refers to
// If t refers to a template type parameter:
// a. x could be a static int data member
// b. x could be a nested class or typedef
// Dependent name: a name cannot be looked up until the actual template arguments are known
//                 it depends on template parameters

// There has to be a way to tell the compiler that certain names are types and that
// certain names aren't

// The "typename" keyword
// eg. typename t::x *f;

// The "template" keyword
// eg. t::template f<int>(), this->template f<int>(); a.template f<int>();

// There categories for template parameters
// 1. Type template parameter T
// 2. None-type template parameter N
// 3. Cast to type template parameter

#include <boost/function.hpp>
#include <iostream>
using namespace std;

struct SumAvg
{
    void operator()(int *values, int n, int &sum, float &avg)
    {

        cout << "SumAvg\n";
        sum = 0;
        for (int i = 0; i < n; i++)
            sum += values[i];
        avg = (float)sum / n;
    };
};

void do_sum_avg(int values[], int n, int &sum, float &avg)
{
    cout << __FUNCTION__ << endl;
    sum = 0;
    for (int i = 0; i < n; i++)
        sum += values[i];
    avg = (float)sum / n;
}

struct A
{
    template <typename T>
    T tmp()
    {
        cout << "template function\n";
    }
};

int main()
{
    boost::function<void(int values[], int n, int &sum, float &avg)> sum_avg;
    sum_avg = do_sum_avg;

    int values[3] = {1, 2, 3};
    int sum;
    float avg;
    sum_avg(values, 3, sum, avg);

    sum_avg = SumAvg();
    sum_avg(values, 3, sum, avg);

    A a;
    a.template tmp<int>(); // template keywords says tmp() is a template function
    return 0;
}