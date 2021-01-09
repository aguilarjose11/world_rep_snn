#include <armadillo>
#include <vector>

#include "lfi.h"
#include "base_snn.h"
#include "snn_ex.h"

#pragma once

/**
 * Euclidean distance matrix
 * 
 * Creates a distance matrix for a given neural network with given distance unit.
 * 
 * @param snn_net a 2d-hidden-layer neural network
 * @param distance_unit
*/
arma::Mat<double> euclidean_distance_matrix(BaseSNN *snn_net, double distance_unit);

/**
 * Euclidean distance matrix
 * 
 * Creates a distance matrix for a given neural network with given distance unit.
 * 
 * @param snn_net 2-d vector [X/Y, point]
 * @param distance_unit
*/
arma::Mat<double> euclidean_distance_matrix(std::vector<std::vector<double>> *point_list, double distance_unit);

/**
 * Random Initial Delay Calculator
 * 
 * Creates a random matrix containing delays for n_neuron input neurons for n_delays postsynaptic neurons.
 * These delays are created to be specifically used as delays between input neurons and hidden/proccessing.
 * 
 * @param n_neurons number of input neurons in matrix
 * @param n_delays 
 * @param l_bound
 * @param u_bound
*/
std::vector<arma::Col<double>> initial_delay_vectors(unsigned int n_neurons, unsigned int n_delays, double l_bound, double u_bound);

/**
 * Initial Weight Calculator
 * 
 * Creates a matrix containing initial weights for a neural network using a distance matrix.
 * @param distance_matrix Distance matrix to use
 * @param sigma_1 Initial weight distribution constant #1
 * @param sigma_2 Initial weight distribution constant #2
*/
std::vector<arma::Col<double>> initial_weight_euclidean(arma::Mat<double> distance_matrix, double sigma_1, double sigma_2);

/**
 * Point map Function
 * 
 * Creates a list of points for a map of dimensions x and y.
 * [dimension (x or y), nth point]
*/
std::vector<std::vector<double>> euclidean_point_map(unsigned int x, unsigned int y);

/**
 * Neural Network Training Class
 * 
 * This class takes care of training a Spiking neural network 
 * using sets of data points.
*/
class SNN
{
    public:
    // TODO: Define constructors before creating code!!!
        SNN() = default;
        /**
         * Spiking Neural Network Trainning constructor. Any parameter marked
         * as (hp) is considered a training hyperparameter.
         * 
         * The constructor will take care of initializing a SNN. Will take care
         * of any math and calculations required for the constructors of the spiking
         * neural nets.
         * 
         * @param i_layer_size Number of input neurons
         * @param h_layer_size Number of neurons in hidden layer
         * @param tau_m Constant of decay for post-synaptic potential 
         * @param u_rest Constant potential of neuron
         * @param init_v Initial Threshold
         * @param t_reset Refractory time after action potential
         * @param k_nought Height of post-synaptic potential
         * @param round_zero Minimum value before post-synaptic potential is treated as zero
         * @param alpha First-Spike-Time scalling factor
         * @param n_x X dimension maximum value
         * @param n_y Y dimension maximum value
         * @param neural_distance Unit distance between neurons (neighbor function)
         * @param distance_unit distance between neurons (spatial distance matrix)
         * @param sigma_1 Initial weight distribution constant #1
         * @param sigma_2 Initial weight distribution constant #2 ()
         * @param l_bound Lower bound for initial delay vector's values
         * @param u_bound Upper bound for initial delay vector's values
         * @param sigma_neighbor Neighborhood function's sigma parameter (HP)
         * @param tau_alpha Weight excitatory change constant (hp)
         * @param tau_beta weight inhibitory change constant (hp)
         * @param eta_w Weight learning step constant (hp)
         * @param eta_d delay learning step constant (hp)
         * @param t_max Maximum time of run for each sample. (hp)
         * @param t_delta time after winner spike considered causatory. (hp)
         * @param ltd_max maximum long term depression applied. (hp)
        */
        SNN(unsigned int i_layer_size, unsigned int h_layer_size, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, unsigned int n_x, unsigned int n_y, double neural_distance, 
        double distance_unit, double sigma_1, double sigma_2, double l_bound, double u_bound,
        double sigma_neighbor, double tau_alpha, double tau_beta, double eta_w, double eta_d,
        unsigned int t_max, unsigned int t_delta, double ltd_max);

        /**
         * Training function
         * 
         * This function will take care of the training of a spiking neural net.
         * The algorithm used is spike time-dependan plasticity (STDP).
         * 
         * @param X Data to be used.
        */
       void train(std::vector<std::vector<double>> X);

        /**
         * General Neighborhood function
         * 
         * Limits the amount of change based on distance between neuron M and N
        */
       double neighbor(SpikeResponseModelNeuron m, SpikeResponseModelNeuron n);

        /**
         * Delay updater.
         * 
         * Using the winner neuron (already found inside network), updates the delays.
        */
        void update_delays(std::vector<double> sample);

        /**
         * Excitatory weight updating function
         * 
         * This function takes care of updating the excitatory synapse weights for a network
         * where at least one neuron fired.
         * 
         * @param network copy of SNN with fired neurons deactivated
         * @param fired_neurons pointer to list to add all fired neurons
         * @param t_epoch Maximum length of time for a new spike to be considered caused by previously-fired neuron.
         * @param sample Data to be used.
         * @param fired_neuron index number of fired neuron
        */
        void update_weights(BaseSNN network, std::vector<unsigned int> *fired_neurons, unsigned int t_epoch, std::vector<double> sample, unsigned int fired_neuron);

        /**
         * Inhibitory weight updating function
         * 
         * This function takes care of updating the neurons given in a list to be considered as inhibitory connections.
         * 
        */
        void update_inhibitory_weights(std::vector<unsigned int> *updated_synapses);


        // Algorithm hyperparameters
        double sigma_neighbor;
        double tau_alpha;
        double tau_beta;
        double eta_w;
        double eta_d;
        unsigned int t_max;
        unsigned int t_delta;
        double ltd_max;

        // spatial distance matrix
        arma::Mat<double> distance_matrix;
        std::vector<std::vector<double>> point_map;

        // spiking neural network
        BaseSNN *snn;









        
};