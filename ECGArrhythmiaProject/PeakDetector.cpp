#include "PeakDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace std;

vector<double> PeakDetector::normalize(const vector<double>& signal) {
    vector<double> result = signal;
    if (result.empty()) return result;

    double mean = accumulate(result.begin(), result.end(), 0.0) / result.size();

    double var = 0.0;
    for (double x : result) var += (x - mean) * (x - mean);
    var /= result.size();

    double stdDev = sqrt(var);
    if (stdDev == 0.0) return result;

    for (double& x : result) x = (x - mean) / stdDev;
    return result;
}

vector<double> PeakDetector::smooth(const vector<double>& signal, int windowSize) {
    if (windowSize <= 1 || signal.empty()) return signal;

    vector<double> smoothed(signal.size());
    int half = windowSize / 2;

    for (size_t i = 0; i < signal.size(); i++) {
        double sum = 0.0;
        int count = 0;

        int start = max(0, static_cast<int>(i) - half);
        int end = min(static_cast<int>(signal.size()) - 1, static_cast<int>(i) + half);

        for (int j = start; j <= end; j++) {
            sum += signal[j];
            count++;
        }

        smoothed[i] = sum / count;
    }

    return smoothed;
}

// ------------------------------------------------------------
// Pan-Tompkins-style QRS detection
// ------------------------------------------------------------

// Low-pass filter.
// Removes some high-frequency noise while preserving the QRS complex.
static vector<double> lowPassFilter(const vector<double>& signal,
                                    double samplingRate) {
    vector<double> output(signal.size(), 0.0);

    if (signal.empty()) return output;

    double dt = 1.0 / samplingRate;
    double cutoffFrequency = 15.0;

    double rc = 1.0 / (2.0 * M_PI * cutoffFrequency);
    double alpha = dt / (rc + dt);

    output[0] = signal[0];

    for (size_t i = 1; i < signal.size(); i++) {
        output[i] =
            output[i - 1] +
            alpha * (signal[i] - output[i - 1]);
    }

    return output;
}


// High-pass filter.
// Reduces slow baseline wander.
static vector<double> highPassFilter(const vector<double>& signal,
                                     double samplingRate) {
    vector<double> output(signal.size(), 0.0);

    if (signal.empty()) return output;

    double dt = 1.0 / samplingRate;
    double cutoffFrequency = 5.0;

    double rc = 1.0 / (2.0 * M_PI * cutoffFrequency);
    double alpha = rc / (rc + dt);

    for (size_t i = 1; i < signal.size(); i++) {
        output[i] =
            alpha *
            (output[i - 1] + signal[i] - signal[i - 1]);
    }

    return output;
}


// Approximate the slope of the ECG.
// QRS complexes normally have a large, rapid change in amplitude.
static vector<double> derivativeFilter(const vector<double>& signal) {
    vector<double> derivative(signal.size(), 0.0);

    for (size_t i = 1; i < signal.size(); i++) {
        derivative[i] = signal[i] - signal[i - 1];
    }

    return derivative;
}


// Squaring makes all values positive and emphasizes large slopes.
static vector<double> squareSignal(const vector<double>& signal) {
    vector<double> squared(signal.size(), 0.0);

    for (size_t i = 0; i < signal.size(); i++) {
        squared[i] = signal[i] * signal[i];
    }

    return squared;
}


// Moving-window integration estimates the energy contained in a QRS-sized
// portion of the signal.
static vector<double> movingWindowIntegration(
    const vector<double>& signal,
    double samplingRate) {

    // Pan-Tompkins-style integration window of approximately 150 ms.
    int windowSize =
        max(1, static_cast<int>(0.150 * samplingRate));

    return PeakDetector::smooth(signal, windowSize);
}


vector<int> PeakDetector::findRPeaks(const vector<double>& signal,
                                     double samplingRate) {

    vector<int> peaks;

    if (signal.size() < 5 || samplingRate <= 0.0) {
        return peaks;
    }

    // --------------------------------------------------------
    // STEP 1: Normalize
    // --------------------------------------------------------
    vector<double> normalized = normalize(signal);


    // --------------------------------------------------------
    // STEP 2: Band-pass-style filtering
    // --------------------------------------------------------
    vector<double> lowPassed =
        lowPassFilter(normalized, samplingRate);

    vector<double> filtered =
        highPassFilter(lowPassed, samplingRate);


    // --------------------------------------------------------
    // STEP 3: Derivative
    // --------------------------------------------------------
    vector<double> derivative =
        derivativeFilter(filtered);


    // --------------------------------------------------------
    // STEP 4: Squaring
    // --------------------------------------------------------
    vector<double> squared =
        squareSignal(derivative);


    // --------------------------------------------------------
    // STEP 5: Moving-window integration
    // --------------------------------------------------------
    vector<double> integrated =
        movingWindowIntegration(squared, samplingRate);


    // --------------------------------------------------------
    // STEP 6: Adaptive threshold
    //
    // Instead of using the old fixed 90th-percentile threshold,
    // continuously estimate signal and noise levels.
    // --------------------------------------------------------

    size_t initializationSamples =
        min(
            integrated.size(),
            static_cast<size_t>(2.0 * samplingRate)
        );

    double initialMean =
        accumulate(
            integrated.begin(),
            integrated.begin() + initializationSamples,
            0.0
        ) / initializationSamples;

    double signalLevel = initialMean * 2.0;
    double noiseLevel = initialMean * 0.5;

    double threshold =
        noiseLevel +
        0.25 * (signalLevel - noiseLevel);


    // Prevent the same QRS complex from being counted twice.
    int refractoryPeriod =
        max(1, static_cast<int>(0.200 * samplingRate));

    // Search around the integrated QRS candidate for the actual
    // R-wave location in the filtered ECG.
    int searchRadius =
        max(1, static_cast<int>(0.280 * samplingRate));

    int lastPeak = -refractoryPeriod;


    for (int i = 1;
         i < static_cast<int>(integrated.size()) - 1;
         i++) {

        bool localMaximum =
            integrated[i] > integrated[i - 1] &&
            integrated[i] >= integrated[i + 1];

        if (!localMaximum) {
            continue;
        }


        if (integrated[i] >= threshold &&
            i - lastPeak >= refractoryPeriod) {

            int start =
                max(0, i - searchRadius);

            int end =
                min(
                    static_cast<int>(filtered.size()) - 1,
                    i + searchRadius
                );


            // Find the largest absolute ECG deflection near the
            // detected QRS complex. Using absolute amplitude also
            // allows inverted ECG leads.
            int rIndex = start;
            double largestAmplitude =
                abs(filtered[start]);

            for (int j = start + 1; j <= end; j++) {

                double amplitude =
                    abs(filtered[j]);

                if (amplitude > largestAmplitude) {
                    largestAmplitude = amplitude;
                    rIndex = j;
                }
            }


            if (peaks.empty() ||
                rIndex - peaks.back() >= refractoryPeriod) {

                peaks.push_back(rIndex);
                lastPeak = rIndex;

                // Update estimated QRS signal level.
                signalLevel =
                    0.125 * integrated[i] +
                    0.875 * signalLevel;
            }

        } else {

            // Candidate was more likely noise.
            noiseLevel =
                0.125 * integrated[i] +
                0.875 * noiseLevel;
        }


        // Continuously update threshold.
        threshold =
            noiseLevel +
            0.25 * (signalLevel - noiseLevel);
    }


    return peaks;
}