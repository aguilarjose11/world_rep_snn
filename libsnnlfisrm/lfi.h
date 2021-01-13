/**
 * Leaky Fire and Integrate model basic code.
 * 
 * This header contains the structure and parts of a Leaky Fire and
 * Integrate neuron.
 * 
 * @author Jose E. Aguilar Escamilla
 * @date January 8th, 2021.
 **/
#pragma once

#include <armadillo>

#define DEBUG true
#define ACTION_POTENTIAL (SHRT_MAX) - 1
#define K_LIST_INPUT_SYNAPSES 0
#define K_LIST_LATERAL_SYNAPSES 1
#define NO_WINNER_SPIKE (UINT_MAX) - 1


/**
 * Neural Spike
 * 
 * Used as the basic unit of information transmition within a 
 * neural network.
 * 
 * The spike's shape is of no interest, but rather its timing,
 * which is timed by the neural network that uses these spikes
*/
class Spike
{
    public:
        bool signal;
};


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
class DelayedSpike: public Spike
{
    public:
        DelayedSpike();
        DelayedSpike(unsigned int delay, bool signal);
        unsigned int delay;
};


/**
 * Leaky Fire-and-Integrate Spike Response Model (SRM) 
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
class SpikeResponseModelNeuron
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
         * @param u_max The maximum PSP/Membrane potential (during spike event)
        */
        SpikeResponseModelNeuron(unsigned int snn_id, int n_inputs, int n_lateral, 
        arma::Col<double> init_d, arma::Col<double> init_w, double tau_m, 
        double u_rest, double init_v, unsigned char t_reset,
        double kappa_naugh, double round_zero, double x, double y, double u_max);

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

        /**
         * This membrane potential temporal variable will contain the 
         * current membrane potential at time of discharge. This is 
         * used because when the neuron fires, it gets rid of any 
         * kappas that may be in wait. This effectivelly gets rid of
         * any posibility of seen the membrane potential until refractory
         * is over. we use a -1 to indicate when this value is not being used.
        */
        int tmp_u;

        /**
         * This is the maximum height of a spike by a neuron. This is reached
         * when the neuron finally spikes. It reaches this value only for a
         * time of 1 t.
        */
        double u_max;

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
        std::vector<DelayedSpike> horizontal_dendrite;


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
        std::vector<DelayedSpike> dendrite;
        
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
        DelayedSpike axon;
};

/**
 * First-spike-time encoding neuron.
 * 
 * Follows the concepts in "Neural Dynamics".
 * The pourpose of this neuron is that of an input layer neuron. It
 * will convert the numerical input into a spike that will occur at
 * time t as Park, et al explain in their paper "T2FSNN: Deep Spiking
 * Neural Networks with Time-to-First-Spike Coding".
 * 
*/
class FirstSpikeTimeNeuron
{
    public:
        /**
         * Fist Spike Time Encoding Neuron
         * 
         * @param snn_id neuron's id within layer
         * @param alpha scaling factor
        */
        FirstSpikeTimeNeuron(unsigned int snn_id, double alpha);

        /**
         * Function that will synchronize the neuron with the
         * rest of the network. It makes the neuron make any
         * calculations that need to get done.
        */
        void t_pulse();

        /**
         * Encoding function
         * 
         * This function will encode the given stimulus using
         * Time-to-First-Spike encoding. Once this function is
         * called, it will set the neuron as ready to fire and 
         * everytime the t_pulse function is called, it will
         * decrease the time left before the spike is sent.
         * 
         * Once a spike is finally sent, the neuron inhibits itself
         * until a new stimulus is given to be encoded.
         * 
         * The value to be encoded will be put in the dendrite member
         * variable. t_pulsing will not touch this variable any more.
        */
        void encode();

        /**
         * Resets the neuron back to its starting state
        */
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
        DelayedSpike axon;

        /**
         * Undefined time.
         * 
         * Used as a flag for when the neuron is inhibited and awaiting
         * a new stimulus.
        */
        const unsigned int UNDEFINED_TIME = (UINT_MAX) - 2;
};

