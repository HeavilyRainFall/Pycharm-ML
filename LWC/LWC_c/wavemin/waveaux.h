/*
Copyright (c) 2014, Rafat Hussain
Copyright (c) 2016, Holger Nahrstaedt
*/
#ifndef WAVEAUX_H_
#define WAVEAUX_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif


int filtlength(const char* name);

int filtcoef(const char* name, float *lp1, float *hp1, float *lp2, float *hp2);

void copy_reverse(const float *in, int N, float *out);

void qmf_even(const float *in, int N, float *out);

void qmf_wrev(const float *in, int N, float *out);

void copy(const float *in, int N, float *out);

int wmaxiter(int sig_len, int filt_len);

void conv_direct(float *inp1, int N, float *inp2, int L, float *oup);

#ifdef __cplusplus
}
#endif


#endif /* WAVEAUX_H_ */
