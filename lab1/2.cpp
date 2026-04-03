#include <iostream>
#include <windows.h>
#include <cmath>
using namespace std;
long long int n;
long long int a[2200000000];
void init()
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
}
void normal()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    unsigned long long int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "normal:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
void multi()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    long long int sum = 0;
    long long int sum1 = 0;
    long long int sum2 = 0;
    int i = 0;
    for (; i < n - 1; i += 2)
    {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
    // 数据肯定能整除2，不需要进行剩余数据额外处理
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "multi:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
void unroll()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    long long int sum = 0;
    long long int sum1 = 0;
    long long int sum2 = 0;
    long long int sum3 = 0;
    long long int sum4 = 0;
    int i = 0;
    for (; i < n - 3; i += 4)
    {
        sum1 += a[i];
        sum2 += a[i + 1];
        sum3 += a[i + 2];
        sum4 += a[i + 3];
    }
    sum = sum1 + sum2 + sum3 + sum4;
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "unroll:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
int main()
{
    while (cin >> n)
    {
        n = pow(2, n);
        init();
        normal();
        multi();
        unroll();
    }
    return 0;
}