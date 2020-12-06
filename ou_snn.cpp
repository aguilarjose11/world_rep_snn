#include <math.h>
#include <exception>
#include "ou_snn.h"

#define ACTION_POTENTIAL (SHRT_MAX) - 1
#define DEBUG true

// Exceptions
class neuronexception: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Invalid neuron initial values";
  }
} neuronex;

class InputLayerException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Invalid inputs for input layer";
  }
} ilex;


D_Spike::D_Spike(unsigned int delay, bool signal)
{
    this->d = delay;
    this->signal = signal;
}

D_Spike::D_Spike()
{
    this->d = 0;
    this->signal = false;
}

/* OU_LIF_SRM::KappaFilter */
OU_LIF_SRM::KappaFilter::KappaFilter(double tau_m, double kappa_nought, 
double min_val, int delay)
{
    this->tau_m = tau_m;
    this->kappa_nought = kappa_nought;
    this->min_val = min_val;
    this->k = kappa_nought;
    this->near_zero = false;
    this->t = -delay - 1;
    this->t_pulse();
}

bool OU_LIF_SRM::KappaFilter::is_zero()
{
    return this->near_zero;
}

double OU_LIF_SRM::KappaFilter::kappa(unsigned int s)
{
    return kappa_nought * (1/exp( s / this->tau_m));
}

void OU_LIF_SRM::KappaFilter::t_pulse()
{
    // move in time forward.
    (this->t)++;
    // make sure this spike is ready to work!
    if(this->t >= 0)
    {
        // update the kappa value
        this->k = this->kappa(this->t);
        // update wether k is low enough to be zero
        if(fabs(this->k) < this->min_val)
        {
            this->near_zero = true;
        }
    }
    else
    {
        this->k = 0;
    }
    
}
/* End of OU_LIF_SRM::KappaFilter */


/* OU_LIF_SRM */

OU_LIF_SRM::OU_LIF_SRM(unsigned int snn_id, int n_inputs, int n_lateral, 
        arma::Col<double> init_d, arma::Col<double> init_w, double tau_m, 
        double u_rest, double init_v, unsigned char t_reset,
        double kappa_naugh, double round_zero)
{   
    // run checks
    if(n_inputs != (int)init_d.size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%d} - {%d}\n", n_inputs, (int)init_d.size());
        throw neuronex;
    }
    if(n_lateral != (int)init_w.size())
    {
        // invalid initial weights
        fprintf(stderr, "invalid initial weights\n");
        throw neuronex;
    }
    if(tau_m <= 0)
    {
        // invalid tau_m
        fprintf(stderr, "invalid tau_m\n");
        throw neuronex;
    }
    if(init_v <= 0)
    {
        // invalid initial threshold
        fprintf(stderr, "invalid initial threshold\n");
        throw neuronex;
    }
    if(t_reset > 200)
    {
        // warn for possible data loop around
        fprintf(stderr, "Warning: possible data loop around\n");
    }
    if(kappa_naugh <= 0)
    {
        // invalid kappa_naugh
        fprintf(stderr, "invalid k_0\n");
        throw neuronex;
    }
    if(round_zero <= 0)
    {
        // invalid round to zero threshold
        fprintf(stderr, "invalid round to zero threshold\n");
        throw neuronex;
    }
    

    this->snn_id = snn_id;
    this->n_inputs = n_inputs;
    this->n_lateral = n_lateral;
    this->d_j = init_d;
    this->w_m = init_w;
    this->tau_m = tau_m;
    this->u_rest = u_rest;
    this->v = init_v;
    this->t_reset = t_reset;
    this->kappa_naugh = kappa_naugh;
    this->round_zero = round_zero;

    // construct neuron's particulars
    this->t = 0;
    this->fired = false;
    this->t_ref = -1;
    // fill in with empty k lists
    // we have input and lateral kappas
    this->k_filter_list.resize(2);

    // initialize the endrites and axons.
    this->horizontal_dendrite.resize(n_lateral);
    this->dendrite.resize(n_inputs);
    this->axon = D_Spike();
}

double OU_LIF_SRM::membrane_potential()
{

    // we can find the membrane potential at current time by gettinh
    // the already-calculated kappa value from the kappa filters.

    // REMEMBER: by the time this neuron recieves a spike from an input
    // synapse, the delay was already applied beforehand. The neuron does
    // not keep track of the time left for the spike to happen. 
    // once it arrives, IT ARRIVES.

    // Sum of all membrane potentials from input layer at time t
    double k_sigma = 0;

    // membrane potentials of input synapses
    for(unsigned int k_i = 0; k_i < this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).size(); k_i++)
    {
        k_sigma += this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(k_i).k;
    }

    // membrane potentials of lateral synapses
    for(unsigned int k_m = 0; k_m < this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).size(); k_m++)
    {
        k_sigma += this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).at(k_m).k;
    }
    

    // the sum of all kappas plus u_rest is the membrane potential.
    return k_sigma + this->u_rest;
}

void OU_LIF_SRM::t_pulse()
{
    if(DEBUG)
        printf("\n\nPulse at time: %u\n", (unsigned int)this->t);
    // march forward
    (this->t)++;
    // count to see if we reach the end of refactory period
    (this->t_ref)++;
    // have we ended refractory period?
    if(this->fired && this->t_ref == this->t_reset)
    {
        // ended refactory period!
        // not refactory period
        this->fired = false;
        // reset the value of t_ref back to -1
        this->t_ref = -1;
    }
    else if(this->fired)
    {
        /**
         * Traditionally, the refactory period would have a function eta
         * to model its behaviour. This behaviour is of decay after the
         * apex of a spike happens and its very steep. Because of the 
         * design of this neural model, no calculations are made for the
         * membrane potential during this time. instead, the neuron does
         * not fire (axon has non-fire spike) and does not listen to any
         * spikes that have a delay smaller than the time left before the
         * neuron exits the refactory period. All existing spikes disapear
         * but again, only spikes that arrive after the refactory period
         * is over are "listened" to.
        */
        // Refactory period
        // create no spike
        this->axon = D_Spike();
        // we go back
        // at this point all of the kappas that existed were removed.
        return;

    }
    // no refractory period.

    // t_pulse all filters forward
    // also remove invalid near zero kappas
    for(unsigned int input = 0; input < k_filter_list.size(); input++)
    {
        for(unsigned int j = 0; j < k_filter_list.at(input).size(); j++)
        {
            k_filter_list.at(input).at(j).t_pulse();
            if(k_filter_list.at(input).at(j).is_zero())
            {
                // we remove this (jth) kappa. it is basically depleted
                k_filter_list.at(input).erase(k_filter_list.at(input).begin()+j);
                j--;
            }
        }
    }

    // check the inputs from input layer to create kappas
    for (unsigned int input = 0; input < this->n_inputs; input++)
    {
        if(this->dendrite.at(input).signal)
        {
            // this is an active spike
            if(!this->fired)
            {
                // Spike arrived. add new kappa.
                if(DEBUG)
                    printf("Spike Recieved.\n");
                KappaFilter new_spike(this->tau_m, this->kappa_naugh, this->round_zero, 0);
                this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).push_back(new_spike);
            }
            // if neuron is in refactory, ignore any spike. DELAYS ALREADY
            // APPLIED!
        }
    }
    // check for horizontal inputs
    for(unsigned int lateral = 0; lateral < this->n_lateral; lateral++)
    {
        // for every lateral synapse
        if(this->snn_id == lateral)
        {
            // no self-feedback
            continue;
        }
        if(this->horizontal_dendrite.at(lateral).signal)
        {
            if(!this->fired)
            {
                // lateral spike arrived, apply the weight!
                double weighted_k = this->kappa_naugh * this->w_m.at(lateral);
                KappaFilter new_spike(this->tau_m, weighted_k, this->round_zero, 0);
                this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).push_back(new_spike);
            }
        }
    }

    // solve the neuron
    if(!this->fired)
    {
        this->solve();
    }
    if(DEBUG)
    {
        printf("input kappas: [");
        for(unsigned int x = 0; x < this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).size(); x++)
        {
            printf(" %f ", this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(x).k);
        }
        printf(" ]\n");
        printf("lateral kappas: [");
        for(unsigned int x = 0; x < this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).size(); x++)
        {
            printf(" %f ", this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).at(x).k);
        }
        printf(" ]\n");
    }

    // anything else?
    

}

void OU_LIF_SRM::solve()
{
    if(!this->fired)
    {
        if(fire_condition(membrane_potential()))
        {
            // we fire!
            this->fired = true;
            this->t_ref = 0;
            // we remove all active kappas!
            this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).clear();
            this->k_filter_list.at(K_LIST_LATERAL_SYNAPSES).clear();
            // we fire the neuron
            D_Spike fire_spike(0, 1);
            this->axon = fire_spike;
        }
        else
        {
            // no firing
            D_Spike no_fire_spike(0, 0);
            this->axon = no_fire_spike;
        }
    }
    
}

bool OU_LIF_SRM::fire_condition(double u)
{
    return u >= this->v;
}

/* end of OU_LIF_SRM */


/* OU_FSTN */

OU_FSTN::OU_FSTN(unsigned int snn_id, double alpha)
{
    // run checks
    if(alpha < 0)
    {
        // invalid alpha
        fprintf(stderr, "Invalid alpha value.\n");
        throw neuronex;
    }

    // construct neuron
    this->snn_id = snn_id;
    this->alpha = alpha;
    this->t = 0;
    this->spike_delay = 0;
    this->dendrite = 0;
    this->axon = D_Spike();
}

void OU_FSTN::t_pulse()
{
    // march forward.
    (this->t)++;
    // update the time delay
    this->spike_delay = (unsigned int) abs(this->alpha * this->dendrite);
    // check requirements to send action potential
    if(this->t % this->spike_delay == 0)
    {
        // Action potential
        /**
         * Note: The action potential has a delay time of zero because as
         * it leaves the cell, the distance between the neuron and the
         * processing layer's targets neurons can be significantly 
         * different. A delay is applied to the spike once it enters the 
         * network space.
        */
        this->axon = D_Spike(0, true);
        if(DEBUG)
            printf("Spike sent\n");
    }
    else
    {
        // No action potential
        if(DEBUG)
            printf("No spike\n");
        this->axon = D_Spike();
    }
    
}


OU_SRM_NET::OU_SRM_NET(unsigned int i_layer_size, unsigned int h_layer_size, 
        arma::Col<double> d_init, arma::Col<double> w_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha)
{
    // run checks
    if(i_layer_size != d_init.size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%u} - {%u}\n", i_layer_size, d_init.size());
        throw neuronex;
    }
    if(h_layer_size != w_init.size())
    {
        // invalid initial weights
        fprintf(stderr, "invalid initial weights\n");
        throw neuronex;
    }
    if(tau_m <= 0)
    {
        // invalid tau_m
        fprintf(stderr, "invalid tau_m\n");
        throw neuronex;
    }
    if(init_v <= 0)
    {
        // invalid initial threshold
        fprintf(stderr, "invalid initial threshold\n");
        throw neuronex;
    }
    if(t_reset > 200)
    {
        // warn for possible data loop around
        fprintf(stderr, "Warning: possible data loop around\n");
    }
    if(k_nought <= 0)
    {
        // invalid kappa_naugh
        fprintf(stderr, "invalid k_0\n");
        throw neuronex;
    }
    if(round_zero <= 0)
    {
        // invalid round to zero threshold
        fprintf(stderr, "invalid round to zero threshold\n");
        throw neuronex;
    }
    
    // initialize variable
    this->t = 0;
    this->i_size = i_layer_size;
    this->h_size = h_layer_size;
    this->d_ji.resize(i_layer_size);
    
    // Populate layers with neurons
    for(int i = 0; i < i_layer_size; i++)
    {
        // create FSTN layer (input)
        this->input_layer.push_back(OU_FSTN(i, alpha));
    }
    for(int h = 0; h < h_layer_size; h++)
    {
        // create hidden layer
        this->hidden_layer.push_back(OU_LIF_SRM(h, i_layer_size, 
        h_layer_size, d_init, w_init, tau_m, u_rest, init_v, t_reset,
        k_nought, round_zero));
    }
    
    // initialize network of spikes between input and hidden layers.
    net_queue_i.reserve(i_layer_size);
    for(int i = 0; i < i_layer_size; i++)
    {
        // for every input network connections
        // reserve a queue for every processing neuron
        net_queue_i.at(i).reserve(h_layer_size);
    }

    // initialize network of spikes between neurons in hidden layer
    net_queue_m.reserve(h_layer_size);
}

void OU_SRM_NET::process(std::vector<double> data)
{
    // make sure we have enough data
    if(data.size() != input_layer.size())
    {
        // no enough data
        fprintf(stderr, "No enough data passed\n");
        throw ilex;

    }
    (this->t)++;

    // insert data into dendrides of input cells, pulse, and move into net
    for(int j = 0; j < i_size; j++)
    {
        input_layer.at(j).dendrite = data.at(j);
        input_layer.at(j).t_pulse();
        // if spikes are outputed, add delay to spike and place in queue
        // for each processing neuron in hiden layer
        if(input_layer.at(j).axon.signal)
        {
            // add delay and add to every hidden neuron
            for(int i = 0; i < h_size; i++)
            {
                // we add +1 to delay to account for current time
                this->net_queue_i.at(j).at(i).push_back(D_Spike(this->d_ji.at(i) + 1, true));
            }
        }
    }

    // Advance time in the network
    for(int j = 0; j < i_size; j++)
    {
        for(int i = 0; i < h_size; i++)
        {
            for(int spike = 0; spike < net_queue_i.at(j).at(i).size(); spike++)
            {
                // send spikes to dendrites of hidden layer
                if(--net_queue_i.at(j).at(i).at(spike).d <= 0)
                {
                    // Spike's delay is over
                    this->hidden_layer.at(i).dendrite.at(j) = net_queue_i.at(j).at(i).at(spike);
                }
                else
                {
                    // spike's delay not over
                    this->hidden_layer.at(i).dendrite.at(j) = D_Spike();
                }
            }
            // advance time in hidden layer
            this->hidden_layer.at(i).t_pulse();

            /**
             * Note for myself:
             * We can add the result of feedback back into the neuron
             * by recalculating the membrane potential
             * 
             * At the same time, it could be a bit more accurate keeping
             * a delay of exactly 1 t?
            */

            // catch the hiden layer's axon and put it in feedback network
            this->net_queue_m.at(i) = this->hidden_layer.at(i).axon;

            /**
             * The neural membrane will be affected until the next epoch?
             * Is it a way to have the feedback affect the neural membrane
             * in this epoch? like a limit? or something?!
            */
        }
    }
    
    


}