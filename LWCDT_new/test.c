/*计算STD*/
static double calcStdWithC(const double* datas, int len)
{
    if(len < 2)
        return 0;
    double sum = 0;
    for(int i = 0; i < len; ++i)
        sum += datas[i];
    double avg = sum / len;
    double diffSum = 0;
    for(int i = 0; i < len; ++i){
        diffSum += (datas[i] - avg) * (datas[i] - avg);
    }
    double std = qSqrt(diffSum / (len - 1));
    return std;
}

/*软阈值*/
static double doThreshold(const double coef, const double thr)
{
    if (fabs(coef) <= thr) {
        return 0;
    }
    if (coef > 0) {
        return coef - thr;
    }
    return coef + thr;
}

/*调用接口*/
void doWaveletTransFormWithC(double* ys, int size)
{
    setbuf(stdout, nullptr);
    const int level = 6; //Decomposition Levels
    wave_object obj = wave_init("db4");// Initialize the wavelet
    wt_object wt = wt_init(obj, "dwt", size, level);// Initialize the wavelet transform object
    setDWTExtension(wt, "sym");// Options are "per" and "sym". Symmetric is the default option
    setWTConv(wt, "direct");
    double* pNewY = new double[size];
    for(int i = 0; i < size; ++i){
        pNewY[i] = ys[i];
    }

    dwt(wt, pNewY);// Perform DWT

    int offset = 0;
    int len = wt->length[0];
    const int cD1lCoefsLen = wt->length[level];
    double* pLastLevelCoefs = new double[cD1lCoefsLen];
    for (int i = 0; i < level; ++i) {
        offset += len;
        len = wt->length[i + 1];
        if(i == level -1){
            for (int j = 0; j < len; ++j) {
                pLastLevelCoefs[j] = wt->output[offset + j];
            }
        }
    }
    int step = cD1lCoefsLen / 10; //将系数分成10份计算std
    int stdOffset = 0;
    double* pStds = new double[cD1lCoefsLen / step + 1];
    for(int i = 0; i < cD1lCoefsLen; i += step){
        int curStep = step;
        if((i + step) > (cD1lCoefsLen - 1))
            curStep = cD1lCoefsLen % step;
        double curStd = calcStdWithC(pLastLevelCoefs +i, curStep);
        if(curStd == 0)
            curStd = pStds[stdOffset - 1];
        pStds[stdOffset++] = curStd;
    }

    double cd1Std = calcStdWithC(pStds, cD1lCoefsLen / step + 1);  //第一层细节系数的std

    double cd1StdSum = 0;
    for(int i = 0; i < stdOffset; ++i){
        cd1StdSum += pStds[i];
    }
    double cd1StdAvg = cd1StdSum / stdOffset;
    const double stdCoef = 1.3;
    const int b = 10;
    double thrshhold = qPow(stdCoef * (cd1StdAvg / cd1Std), b);
    if(thrshhold > 1000)
        thrshhold = 1000;
    len = wt->length[0];
    offset = 0;
    for (int i = 0; i < level; ++i) {
        offset += len;
        len = wt->length[i + 1];
        for (int j = 0; j < len; ++j) {
            wt->output[offset + j] = doThreshold(wt->output[offset + j], thrshhold);
        }
    }

    idwt(wt, pNewY);// Perform IDWT
    for(int i = 0; i < size; ++i){
        ys[i] = pNewY[i];
    }

    delete[] pStds;
    delete[] pLastLevelCoefs;
    delete[] pNewY;
    wave_free(obj);
    wt_free(wt);
}
