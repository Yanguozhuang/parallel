#include <iostream>
#include <windows.h>
using namespace std;
int n;
int a[10001];
int b[10001][10001];
int sum[10001];

void init()
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i][j] = i + j;
        }
    }
}

void normal()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int j = 0; j < n; j++)
    {
        sum[j] = 0;
        for (int i = 0; i < n; i++)
        {
            sum[j] += b[i][j] * a[i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "normal:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
void cache()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < n; i++)
    {
        sum[i] = 0;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum[j] += b[i][j] * a[i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "cache:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
void unroll()
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < n; i++)
    {
        sum[i] = 0;
    }
    for (int i = 0; i < n; i++)
    {
        int j = 0;
        for (; j <= n - 4; j += 4)
        {
            sum[j] += b[i][j] * a[i];
            sum[j + 1] += b[i][j + 1] * a[i];
            sum[j + 2] += b[i][j + 2] * a[i];
            sum[j + 3] += b[i][j + 3] * a[i];
        }
        for (; j < n; j++)
        {
            sum[j] += b[i][j] * a[i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "unroll:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}
int main()
{
    while (cin >> n)
    {
        if (n > 10000)
            continue;
        init();
        normal();
        cache();
        unroll();
    }
    return 0;
}