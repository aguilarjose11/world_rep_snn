#include <vector>
#include <armadillo>

#include "lfi.h"

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
        /***/
        BaseSNN(unsigned int i_layer_size, unsigned int h_layer_size, 
        std::vector<arma::Col<double>> d_init, std::vector<arma::Col<double>> w_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, unsigned int n_x, unsigned int n_y, double neural_distance);

        /**
         * Process function
         * 
         * Processes the given data vector and moves the network forward
        */
        void process(std::vector<double> data);

        /**
         * Spike checker
         * 
         * Checks wether there is a winner spike
         * 
         * @return NO_WINNER_SPIKE if no spikes, else, winner neuron's index.
        */
        unsigned int find_winner_spike();

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
         * Reset runction
         * 
         * This function resets a neural network back to its initial state using new delays and weights.
         * 
         * @param init_d initial delays
         * @param init_w initial weight
        */
       void reset(arma::Col<double> init_d, arma::Col<double> init_w);

        // variables
        unsigned long long t;
        unsigned int i_size;
        unsigned int h_size;
        unsigned int n_x;
        unsigned int n_y;
        double neural_distance;
        // is it there a winner spike?
        bool has_winner;
        // winner spike
        unsigned int winner_neuron;

        /**
         * Delay vector for the ith processing neuron in hidden layer
         * [input neuron, hidden layer neuron]
        */
        std::vector<arma::Col<double>> d_ji;
        std::vector<FirstSpikeTimeNeuron> input_layer;
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
        std::vector<std::vector<std::vector<DelayedSpike>>> net_queue_i;

        /**
         * Lateral synapse network queue.
        */
        std::vector<DelayedSpike> net_queue_m;
    // TODO.
};
