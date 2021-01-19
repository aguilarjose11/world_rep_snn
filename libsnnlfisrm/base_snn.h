#include <vector>
#include <armadillo>

#include "lfi.h"

#define NO_WINNER (UINT_MAX) - 1

#pragma once

/**
 * The OU Spike Response Model Neural Network
 * 
 * This network builder will create a network that uses
 * OU_LIF_SRM neurons.
 * 
 * The network would be a multilayered network where there
 * exists input/output and hidden layers. This network is
 * unique in that it can keep track of D_Spikes before sending
 * ay spikes to their respective owners.
*/
class BaseSNN
{
    public: 
        /**
         * Constructor.
        */
        BaseSNN(unsigned int i_layer_size,
        std::vector<std::vector<double>> d_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, double u_max);

        /**
         * Process function
         * 
         * Solves the network with the given current stimulus.
         * 
         * @param None
        */
        void process();


        /**
         * Reset runction
         * 
         * This function resets a neural network back to its initial state. Uses the delays
         * already within the network.
         * 
         * @param None
        */
        void reset();

        /**
         * Re-processing function
         * 
         * Will reset all neurons back to starting, reset the time of the network
         * and store the new given data. 
         * 
         * @param None
        */
        void re_process(std::vector<double> data);

        // variables
        
        // Input layer size
        unsigned int i_layer_size;

        // Processing layer size
        unsigned int h_layer_size;

        // Current time
        unsigned long long t;

        // Winner neuron. NO_WINNER if none, else will contain the id
        unsigned int winner_neuron;

        /**
         * Delay vector for the ith processing neuron in hidden layer
         * [input neuron, hidden layer neuron]
        */
        std::vector<std::vector<double>> d_ji;

        std::vector<std::vector<double>> d_ji_reset;

        // Array containing the input neurons
        std::vector<FirstSpikeTimeNeuron> input_layer;

        // Array containing the processing neurons
        std::vector<SpikeResponseModelNeuron> hidden_layer;

        /**
         * Input to Processing layer queue
         * 
         * These spikes are delayed by some value and depends
         * on the 
         * index order: 
         * [input neuron, processing neuron, place in neuron queue]
         * 
         * The number of spikes in neuron queue can vary.
        */
        std::vector<std::vector<std::vector<DelayedSpike>>> queue_ji;

        /**
         * Stimulus being processed
        */
        std::vector<double> stimuli;

        /**
         * flag for newness of stimulus
        */
        bool is_stimulus_new;

    // TODO.
};
