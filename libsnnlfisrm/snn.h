#include <armadillo>
#include <vector>

#include "lfi.h"
#include "base_snn.h"
#include "snn_ex.h"

#define UNDEFINED_TIME (UINT_MAX) - 2

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
 * Creates a delay vector that is equaly distanced and rectangular in shape
 * 
*/
std::vector<arma::Col<double>> initial_delay_vector_2d_map(unsigned int n_x, 
unsigned int n_y, unsigned int delays_per_row, unsigned int delays_per_column);

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
         * @param n_data Number of input neurons & expected dimensions of stimuli
         * @param tau_m Constant of decay for post-synaptic potential 
         * @param u_rest Constant potential of neuron
         * @param init_v Initial Threshold
         * @param t_reset Refractory time after action potential
         * @param k_nought Height of post-synaptic potential
         * @param round_zero Minimum value before post-synaptic potential is treated as zero
         * @param alpha First-Spike-Time scalling factor
         * @param n_x X dimension maximum value
         * @param n_y Y dimension maximum value
         * @param delay_distance Unit distance between neurons (delay function)
         * @param sigma_neighbor Neighborhood function's sigma parameter (HP)
         * @param eta_d delay learning step constant (hp)
         * @param t_max Maximum time of run for each sample. (hp)
         * @param u_max Maximum potential of spike during spiking.
        */
        SNN(unsigned int n_data, double tau_m, double u_rest, double init_v, 
        double t_reset, double k_nought, double round_zero, double alpha, 
        unsigned int n_x, unsigned int n_y, double delay_distance, 
        double distance_unit, double sigma_neighbor, double eta_d,
        unsigned int t_max, double u_max);

        /**
         * Training function
         * 
         * This function will take care of the training of a spiking neural net.
         * The algorithm used is spike time-dependan plasticity (STDP).
         * 
         * The implemented algorithm here is hebbian learning by default.
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




        // Algorithm hyperparameters
        double sigma_neighbor;
        double eta_d;
        unsigned int t_max;

        // spatial distance matrix
        arma::Mat<double> distance_matrix;
        std::vector<std::vector<double>> point_map;

        // spiking neural network
        BaseSNN *snn;









        
};