#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>

// Define the number of hidden states (e.g., Bull and Bear)
const int NUM_STATES = 2;
const int NUM_OBSERVATIONS = 100; // Adjust based on your dataset

// Generate random stock return data (simulating market returns)
std::vector<double> generateReturns(int numDays) {
    std::vector<double> returns(numDays);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Bull market: Mean positive returns, low volatility
    std::normal_distribution<double> bullDist(0.005, 0.01);
    
    // Bear market: Mean negative returns, higher volatility
    std::normal_distribution<double> bearDist(-0.005, 0.02);

    bool isBull = true;
    for (int i = 0; i < numDays; i++) {
        if (i % 50 == 0)  // Regime shift every 50 days (simulated)
            isBull = !isBull;
        returns[i] = isBull ? bullDist(gen) : bearDist(gen);
    }

    return returns;
}

// Initialize the HMM parameters
struct HMM {
    Eigen::MatrixXd transitionMatrix; // State transition probabilities
    Eigen::MatrixXd emissionMeans;    // Emission probabilities
    Eigen::VectorXd initialProbabilities; // Initial state probabilities
};

// Train the HMM using an expectation-maximization (EM) approach
void trainHMM(HMM &hmm, const std::vector<double> &returns) {
    int T = returns.size();

    // Expectation step: Compute probabilities of hidden states given observations
    Eigen::MatrixXd forward(T, NUM_STATES);
    Eigen::MatrixXd backward(T, NUM_STATES);

    // Initialize forward probabilities
    for (int i = 0; i < NUM_STATES; i++) {
        forward(0, i) = hmm.initialProbabilities(i) * hmm.emissionMeans(i, 0);
    }

    // Forward algorithm
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < NUM_STATES; j++) {
            double sum = 0;
            for (int i = 0; i < NUM_STATES; i++) {
                sum += forward(t - 1, i) * hmm.transitionMatrix(i, j);
            }
            forward(t, j) = sum * hmm.emissionMeans(j, 0);
        }
    }

    // Backward algorithm
    for (int i = 0; i < NUM_STATES; i++) {
        backward(T - 1, i) = 1.0;
    }
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < NUM_STATES; i++) {
            double sum = 0;
            for (int j = 0; j < NUM_STATES; j++) {
                sum += hmm.transitionMatrix(i, j) * hmm.emissionMeans(j, 0) * backward(t + 1, j);
            }
            backward(t, i) = sum;
        }
    }

    // Compute state probabilities
    Eigen::MatrixXd gamma(T, NUM_STATES);
    for (int t = 0; t < T; t++) {
        double sum = 0;
        for (int i = 0; i < NUM_STATES; i++) {
            gamma(t, i) = forward(t, i) * backward(t, i);
            sum += gamma(t, i);
        }
        for (int i = 0; i < NUM_STATES; i++) {
            gamma(t, i) /= sum;
        }
    }

    // Maximization step: Update parameters
    for (int i = 0; i < NUM_STATES; i++) {
        hmm.initialProbabilities(i) = gamma(0, i);
    }

    // Update transition probabilities
    for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_STATES; j++) {
            double numerator = 0, denominator = 0;
            for (int t = 0; t < T - 1; t++) {
                numerator += forward(t, i) * hmm.transitionMatrix(i, j) * hmm.emissionMeans(j, 0) * backward(t + 1, j);
                denominator += gamma(t, i);
            }
            hmm.transitionMatrix(i, j) = numerator / denominator;
        }
    }
}

// Predict the market regime
std::vector<int> predictRegime(const HMM &hmm, const std::vector<double> &returns) {
    int T = returns.size();
    std::vector<int> predictedStates(T);

    // Compute likelihoods for each state
    for (int t = 0; t < T; t++) {
        double bestProb = -1.0;
        int bestState = 0;

        for (int i = 0; i < NUM_STATES; i++) {
            double prob = hmm.initialProbabilities(i) * hmm.emissionMeans(i, 0);
            if (prob > bestProb) {
                bestProb = prob;
                bestState = i;
            }
        }
        predictedStates[t] = bestState;
    }

    return predictedStates;
}

int main() {
    // Generate synthetic return data
    std::vector<double> returns = generateReturns(NUM_OBSERVATIONS);

    // Initialize HMM
    HMM hmm;
    hmm.transitionMatrix = Eigen::MatrixXd(NUM_STATES, NUM_STATES);
    hmm.transitionMatrix << 0.9, 0.1,
                            0.2, 0.8; // Higher probability of staying in the same state

    hmm.emissionMeans = Eigen::MatrixXd(NUM_STATES, 1);
    hmm.emissionMeans << 0.005, // Bull market mean return
                         -0.005; // Bear market mean return

    hmm.initialProbabilities = Eigen::VectorXd(NUM_STATES);
    hmm.initialProbabilities << 0.5, 0.5; // Equal initial probability for each state

    // Train HMM
    trainHMM(hmm, returns);

    // Predict market regime
    std::vector<int> predictedStates = predictRegime(hmm, returns);

    // Print results
    std::cout << "Market Regime Predictions (0 = Bear, 1 = Bull):\n";
    for (size_t i = 0; i < predictedStates.size(); i++) {
        std::cout << "Day " << i + 1 << ": " << (predictedStates[i] == 0 ? "Bear" : "Bull") << "\n";
    }

    return 0;
}
