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
         * @param init_w Initial weight vector
         * @param tau_m Membrane time constant
         * @param tau_s PSP time constant
         * @param u_rest Resting potential
         * @param init_v Initial threshold
        */
        OU_LIF_SRM(unsigned int snn_id, arma::Col<double> init_w,
         double tau_m, double tau_s, double u_rest, double init_v,
         double n_inputs, unsigned char t_reset, double kappa_naugh,
         double min_val);

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

            private:
                double tau_m;
                double kappa_nought;
                double min_val;
                int t;
                bool near_zero;
        };


        /**
         * Weight matrix.
         * 
         * Provides the weights for each input spike.
         * It is indexed as j where it is the input
         * presyaptic neuron's weight. 
        */
        arma::Col<double> w_j;

        /**
         * Time tracking variable
        */
        unsigned long long t;

        /**
         * neuron's own index within a layer.
        */
        unsigned int i;

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
         * Spike Threshold
         * 
         * Potential level at which the neuron fires.
        */
        double v;
        double kappa_naugh;
        double tau_m;
        double tau_s;
        double min_val;

        /**
         * Resting potential of the neuron.
        */
        double u_rest;

        /**
         * neuron firing flag
        */
        bool fired;

        /**
         * Time of reset after firing.
        */
        unsigned char t_reset;

        /**
         * Current time in refactory. We reset to -1 as flag that we 
         * have not fired yet
        */
        int t_ref;

        /**
         * variable-size list that will contain kappa filters
         * for each synapse with input. each input will be
         * indexed as their number as input neuron within the
         * input layer.
        */
        
        std::vector< std::vector<KappaFilter> > k_filter_list;
        
};


class OU_LIF_SRM_INPUT
{
    public:
        OU_LIF_SRM_INPUT(double encoding_factor);
        /**
         * 
        */
        void t_pulse();

        /**
         * Encoding function
         * 
         * Uses first-spike-time encoding
        */
        double encode(double x);

        /**
         * encoded rate.
        */
        unsigned int spike_rate;

        /**
         * factor used for encoding
        */
        double encoding_factor;

        /**
         * input value
        */
        double dendrite;

        /**
         * output spike
        */
        D_Spike axon;
};


class OU_SNN_NET
{

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
        OU_SRM_NET(unsigned int n_inputs, unsigned int n_hidden);
        std::vector<OU_LIF_SRM_INPUT> input_layer
        std::vector<OU_LIF_SRM> hidden_layer;
    // TODO.
};