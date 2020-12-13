/**
 * Spiking Neural Network Framework used for the tour
 * robot project for CS-4023-Inteligent Robotics.
 * 
 * @author Jose E. Aguilar Escamilla
 * @date December 1st, 2020
 * @li 
 **/

#pragma once

#include <armadillo>
#include <vector>

#define K_LIST_INPUT_SYNAPSES 0
#define K_LIST_LATERAL_SYNAPSES 1
#define NO_WINNER_SPIKE (UINT_MAX) - 1

/**
 * Delayable Spike
 * 
 * Used in the modified Spike Response Model. It is a spike
 * that can be delayed for an amount of time.
 * 
 * The spikes can form a queue when connected in a network.
 * The delay d will be a countdown variable that once it reaches
 * 0, it is tought to have "arrived" to destination.
*/
class D_Spike
{
    public:
        D_Spike();
        D_Spike(unsigned int delay, bool signal);
        unsigned int d;
        // we can have no spike, or spike.
        bool signal;
};

/**
 * Leaky Fire-and-Integrate -> Spike Response Model (SRM) 
 * 
 * This neural model follows the principles of the SRM
 * model as defined in 'Neural Dynamics' book and the
 * model implemented by Alamdari.
 * 
 * This neuron takes not only inputs from an input layer
 * but also accepts horizontal inputs from neurons in
 * the same layer.
 * 
 * Refer to the citations of this project on more information
 * on the neural models implemented in this work.
 * 
 * On regular models, the refactory filter (eta) is used to
 * describe the membrane's refactory period. In this model,
 * we asume the neuron reaches u_r immediately, but remains
 * on refactory period for a specified (t_reset) amount of time.
 * During this time, no spikes can enter, and the neuron resets
 * all kappa lists to be empty so that when the refactory period
 * ends, the neuron is back to u_rest.
 **/
class OU_LIF_SRM
{
    public:
        /**
         * Constructor for custom neuron.
         * 
         * @param snn_id Neuron's id withing a layer
         * @param n_inputs  number of input synapses
         * @param n_lateral number of lateral synapses
         * @param init_d initial delay vector for input synapses (Deprecated)
         * @param init_w Initial weight vector for lateral synapses
         * @param tau_m membrane behaviour constant. (spike decay)
         * @param u_rest Resting potential
         * @param init_v Initial threshold
         * @param t_rest length of refractory period
         * @param kappa_naugh max height of spike (membrane's)
         * @param round_zero value at which kappa function is zero.
        */
        OU_LIF_SRM(unsigned int snn_id, int n_inputs, int n_lateral, 
        arma::Col<double> init_d, arma::Col<double> init_w, double tau_m, 
        double u_rest, double init_v, unsigned char t_reset,
        double kappa_naugh, double round_zero, double x, double y);

        /**
          * Membrane potential function ( u(t) )
          * 
          * The membrane potential function gives the 
          * potential levels at current time t. This
          * function shall be used for the firing 
          * decision.
          * @param None
          * @return The current membrane potential
          *             
        **/
        double membrane_potential();

        /**
         * Reset runction
         * 
         * This function resets a neuron back to its initial state.
         * 
         * @param None
        */
       void reset();

        /**
         * March-Forward function
         * 
         * Te t_pulse function triggers the solution of the 
         * integration of the cell. it helps time the spike
         * firing and processing. It also creates a time
         * value similar to epoch.
        */
        void t_pulse();

        /**
         * Integration function.
         * 
         * This function takes care of performing the 
         * integration of the inputs, decide wether the
         * neuron needs to be triggered, and applies 
         * the weights and delays for the inputs (not really?).
        */
        void solve();

        /**
         * Firing Condition
         * 
         * The conditions for the membrane to fire.
        */
        bool fire_condition(double u);

        /**
         * Disable function
         * 
         * Used for training. This function disables the activation of the neuron
        */
       void disable();

        /**
         * Membrane Behaviour Class
         * 
         * This class models the behaviour of the
         * cellular membrane of a neuron.
         * 
         * It is designed to be added to a list to
         * account for the multiple spikes that could 
         * come from the same synapse. there will be a
         * list/cvector that will hold these kappa
         * filters. They will be added to get the final 
         * sum of all kappas to account for spikes BEFORE
         * t_i (trigger time)
        */
        class KappaFilter
        {
            public:
                /**
                 * Constructor for individual kappa filter
                 * @param tau_m Decay rate
                 * @param eta_nought Strength of spike (w!)
                 * @param min_val Minimum value before considered zero
                 * @param delay Spike delay (inputs)
                */
                KappaFilter(double tau_m, double kappa_nought, 
                             double min_val, int delay);

                /**
                 * Is the kappa filter near zero?
                */
                bool is_zero();

                /**
                 * Kappa function (Postsynaptic potential)
                 * 
                 * Used to give the individual postsynaptic potential
                 * for the current neuron.
                */
                double kappa(unsigned int s);

                /**
                 * March-Forward Function
                 * 
                 * Function that moves the function ahead and
                 * stores values.
                */
                void t_pulse();

                /**
                 * Kappa filter value. 
                */
                double k;
                double tau_m;
                double kappa_nought;
                double min_val;
                int t;
                bool near_zero;
        };


        // Constructor variables
        /**
         * neuron's own index within a layer.
        */
        unsigned int snn_id;
        unsigned int n_inputs;
        unsigned int n_lateral;
        /**
         * delay vector for input synapses
         * 
         * may disapear in future updates since this is really managed 
         * by the network framework.
        */
        arma::Col<double> d_j;
        /**
         * Weight matrix.
         * 
         * Provides the weights for each input spike.
         * It is indexed as j where it is the input
         * presyaptic neuron's weight. 
        */
        arma::Col<double> w_m;
        double tau_m;
        /**
         * Resting potential of the neuron.
        */
        double u_rest;
        /**
         * Spike Threshold
         * 
         * Potential level at which the neuron fires.
        */
        double v;
        /**
         * Time of reset after firing.
        */
        unsigned char t_reset;
        double kappa_naugh;
        double round_zero;
        double x;
        double y;
        bool disabled;


        /**
         * Time tracking variable
        */
        unsigned long long t;

        /**
         * neuron firing flag
        */
        bool fired;


        /**
         * Current time in refactory. We reset to -1 as flag that we 
         * have not fired yet
        */
        int t_ref;

        // Neuron Anatomy

        /**
         * variable-size list that will contain kappa filters
         * for each synapse with input. each input will be
         * indexed as their number as input neuron within the
         * input layer.
        */
        
        std::vector< std::vector<KappaFilter> > k_filter_list;

        /**
         * Horizontal Input function
         * 
         * The inputs comming from the other neurons on the
         * same layer of neurons is proccessed here. 
         * 
         * The synapses have a delay applied to them when
         * they first come in.
        */
        std::vector<D_Spike> horizontal_dendrite;


        /**
         * Dendrite input
         * 
         * The variable will take a matrix that contains
         * the input spikes for the neural cell. This is
         * performed by the network framework.
         * This only processes the inputs from a previous
         * layer. For processing the side-neural spikes,
         * it is necesary to create a model-specific implementation
         * of this interface.
         * 
         * The inputs comming from the input layer have a delay
         * applied to them.
        */
        std::vector<D_Spike> dendrite;
        
        /**
         * Axon output
         * 
         * The axon in the neuron model represents the
         * place where outcomming spikes come out of the
         * neuron. The spike is either a 1 or a 0.
         * 
         * if the spike in the axon is a 0, no delay is expected
         * else, some delay could be there.
        */
        D_Spike axon;
};

/**
 * First-spike-time encoding neuron.
*/
class OU_FSTN
{
    public:
        /**
         * Fist Spike Time Encoding Neuron
         * 
         * @param snn_id neuron's id within layer
         * @param alpha scaling factor
        */
        OU_FSTN(unsigned int snn_id, double alpha);

        /**
         * 
        */
        void t_pulse();

        void reset();

        unsigned int snn_id;

        double alpha;

        unsigned long long t;

        /**
         * encoded rate.
        */
        unsigned int spike_delay;

        /**
         * input value
        */
        double dendrite;

        /**
         * output spike
        */
        D_Spike axon;
};


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
class OU_SRM_NET
{
    public: 
        /***/
        OU_SRM_NET(unsigned int i_layer_size, unsigned int h_layer_size, 
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
        std::vector<OU_FSTN> input_layer;
        std::vector<OU_LIF_SRM> hidden_layer;

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
        std::vector<std::vector<std::vector<D_Spike>>> net_queue_i;

        /**
         * Lateral synapse network queue.
        */
        std::vector<D_Spike> net_queue_m;
    // TODO.
};

/**
 * Euclidean distance matrix
 * 
 * Creates a distance matrix for a given neural network with given distance unit.
 * 
 * @param snn_net a 2d-hidden-layer neural network
 * @param distance_unit
*/
arma::Mat<double> euclidean_distance_matrix(OU_SRM_NET *snn_net, double distance_unit);

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
class OU_SRMN_TRAIN
{
    public:
    // TODO: Define constructors before creating code!!!
        OU_SRMN_TRAIN() = default;
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
        OU_SRMN_TRAIN(unsigned int i_layer_size, unsigned int h_layer_size, double tau_m,
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
       double neighbor(OU_LIF_SRM m, OU_LIF_SRM n);

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
        void update_weights(OU_SRM_NET network, std::vector<unsigned int> *fired_neurons, unsigned int t_epoch, std::vector<double> sample, unsigned int fired_neuron);

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
        OU_SRM_NET *snn;









        
};