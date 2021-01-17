#include "lfi.h"
#include "snn_ex.h"


/////////////////////////////////////////////////////////////////////
DelayedSpike::DelayedSpike()
{
    this->delay = 0;
    this->signal = false;
}

DelayedSpike::DelayedSpike(unsigned int delay, bool signal)
{
    this->delay = delay;
    this->signal = signal;
}
/////////////////////////////////////////////////////////////////////
SpikeResponseModelNeuron::SpikeResponseModelNeuron(unsigned int snn_id, int n_inputs,
        arma::Col<double> init_d, double tau_m, 
        double u_rest, double init_v, unsigned char t_reset,
        double kappa_naugh, double round_zero, double u_max)
{   
    // run checks
    if(n_inputs != (int)init_d.size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%d} - {%d}\n", n_inputs, (int)init_d.size());
        throw neuronexception();
    }
    if(tau_m <= 0)
    {
        // invalid tau_m
        fprintf(stderr, "invalid tau_m\n");
        throw neuronexception();
    }
    if(init_v <= 0)
    {
        // invalid initial threshold
        fprintf(stderr, "invalid initial threshold\n");
        throw neuronexception();
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
        throw neuronexception();
    }
    if(round_zero <= 0)
    {
        // invalid round to zero threshold
        fprintf(stderr, "invalid round to zero threshold\n");
        throw neuronexception();
    }
    

    this->snn_id = snn_id;
    this->n_inputs = n_inputs;
    this->d_j = init_d;
    this->tau_m = tau_m;
    this->u_rest = u_rest;
    this->v = init_v;
    this->t_reset = t_reset;
    this->kappa_naugh = kappa_naugh;
    this->round_zero = round_zero;
    this->disabled = false;

    // construct neuron's particulars
    this->t = 0;
    this->fired = false;
    this->t_ref = -1;
    this->tmp_u = -1; // -1 = not being used.
    this->u_max = u_max;
    // fill in with empty k lists
    // we have input kappas only!
    this->k_filter_list.resize(1);

    // initialize the endrites and axons.
    this->dendrite.resize(n_inputs);
    this->axon = DelayedSpike();
    // Temporal membrane potential for when the neuron spikes
    // and it removes the kappas.
}

double SpikeResponseModelNeuron::membrane_potential()
{

    // we can find the membrane potential at current time by getting
    // the already-calculated kappa value from the kappa filters.

    // REMEMBER: by the time this neuron recieves a spike from an input
    // synapse, the delay was already applied beforehand. The neuron does
    // not keep track of the time left for the spike to happen. 
    // once it arrives, IT ARRIVES.

    if(this->tmp_u != -1)
    {
        // We need to give the refractory membrane potential (aka. max PSP)
        return tmp_u;
    }

    // Sum of all membrane potentials from input layer at time t
    double k_sigma = 0;

    // membrane potentials of input synapses
    for(unsigned int k_i = 0; k_i < this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).size(); k_i++)
    {
        k_sigma += this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(k_i).k;
    }

    // the sum of all kappas plus u_rest is the membrane potential.
    return k_sigma + this->u_rest;
}

void SpikeResponseModelNeuron::t_pulse()
{
    if(DEBUG)
        printf("\n\nPulse at time: %u\n", (unsigned int)this->t);
    // march forward
    (this->t)++;
    // Make sure that we come down from the maximum spike after spike.
    if(tmp_u != -1)
    {
        tmp_u = -1;
    }
    // count to see if we reach the end of refactory period
    (this->t_ref)++;// what?!
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
        // Refactory period still on going
        // create no spike
        this->axon = DelayedSpike();
        // we go back
        // at this point all of the kappas that existed were removed.
        return;

    }
    // no refractory period.

    // check the inputs from input layer to create kappas
    for (unsigned int input = 0; input < this->n_inputs; input++)
    {
        if(this->dendrite.at(input).signal)
        {

            // Spike arrived. add new kappa.
            if(DEBUG)
                printf("Spike Recieved. Adding kappa.\n");
            KappaFilter new_spike(this->tau_m, this->kappa_naugh, this->round_zero, 0);
            this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).push_back(new_spike);
        }
    }

    // t_pulse all filters forward
    // also remove invalid near zero kappas

    for(unsigned int j = 0; j < k_filter_list.at(K_LIST_INPUT_SYNAPSES).size(); j++)
    {
        k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(j).t_pulse();
        if(k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(j).is_zero())
        {
            // we remove this (jth) kappa. it is basically depleted
            k_filter_list.at(K_LIST_INPUT_SYNAPSES).erase(k_filter_list.at(K_LIST_INPUT_SYNAPSES).begin()+j);
            j--;
        }
    }

    // Solve the neuron.
    this->solve();

    if(DEBUG)
    {
        printf("input kappas: [");
        for(unsigned int x = 0; x < this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).size(); x++)
        {
            printf(" %f ", this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).at(x).k);
        }
        printf(" ]\n");
    }

    // anything else?
    

}

void SpikeResponseModelNeuron::solve()
{
    if(!this->fired && !this->disabled)
    {
        if(fire_condition(membrane_potential()))
        {
            // We meet the fire condition. Refractory period will begin shortly
            // we fire!
            this->fired = true;
            this->t_ref = 0;
            // we remove all active kappas!
            this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).clear();
            // we fire the neuron
            DelayedSpike fire_spike(0, true);
            this->axon = fire_spike;
            this->tmp_u = this->u_max;
        }
        else
        {
            // no firing
            DelayedSpike no_fire_spike(0, 0);
            this->axon = no_fire_spike;
        }
    }
    else
    {
        // no firing
        DelayedSpike no_fire_spike(0, 0);
        this->axon = no_fire_spike;
    }
    
    
}

bool SpikeResponseModelNeuron::fire_condition(double u)
{
    return u >= this->v;
}

void SpikeResponseModelNeuron::disable()
{
    this->disabled = true;
}

void SpikeResponseModelNeuron::reset()
{
    // set time back to 0
    this->t = 0;
    // remove all kappas
    this->k_filter_list.at(K_LIST_INPUT_SYNAPSES).clear();
    // enable neuron
    this->disabled = false;
    // set firing back to no
    this->fired = false;
    // reset refractory timer
    this->t_ref = -1;
    // reset temporal membrane potential
    this->tmp_u = -1;
    // reset axon to no spikes
    this->axon = DelayedSpike();
    // clear input dendrites.
    this->dendrite.clear();
    this->dendrite.resize(this->n_inputs);
    /**
     * By this point, the weights for each connection was probably reset.
    */
}
/////////////////////////////////////////////////////////////////////
SpikeResponseModelNeuron::KappaFilter::KappaFilter(double tau_m, double kappa_nought, 
double min_val, int delay)
{
    this->tau_m = tau_m;
    this->kappa_nought = kappa_nought;
    this->min_val = min_val;
    this->k = kappa_nought;
    this->near_zero = false;
    this->t = -2; // note the delay of 1 between spike and actual effect on psp
    this->t_pulse();
}

bool SpikeResponseModelNeuron::KappaFilter::is_zero()
{
    return this->near_zero;
}

double SpikeResponseModelNeuron::KappaFilter::kappa(unsigned int s)
{
    return kappa_nought * (1/exp( s / this->tau_m));
}

void SpikeResponseModelNeuron::KappaFilter::t_pulse()
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
/////////////////////////////////////////////////////////////////////
FirstSpikeTimeNeuron::FirstSpikeTimeNeuron(unsigned int snn_id, double alpha)
{
    // run checks
    if(alpha < 0)
    {
        // invalid alpha
        fprintf(stderr, "Invalid alpha value.\n");
        throw neuronexception();
    }

    // construct neuron
    this->snn_id = snn_id;
    this->alpha = alpha;
    this->t = 0;
    this->spike_delay = this->UNDEFINED_TIME;
    this->dendrite = 0;
    this->axon = DelayedSpike();
}

void FirstSpikeTimeNeuron::encode()
{
    // encode by finding the delay time before the next spike.
    this->spike_delay = (unsigned int) abs(this->alpha * this->dendrite);
}

void FirstSpikeTimeNeuron::t_pulse()
{
    // march forward.
    (this->t)++;
    // update the time delay
    if(DEBUG)
        printf("Delay to be calculated\n");

    if(this->spike_delay != UNDEFINED_TIME && --(this->spike_delay) <= 0)
    {
        // We have a spike due.
        this->axon = DelayedSpike(0, true);
        if(DEBUG)
            printf("Spike sent\n");
        // Inhibit the neuron until a new stimulus arrives
        this->spike_delay = UNDEFINED_TIME;
        return;
    }
    else 
    {
        this->axon = DelayedSpike(0, false);
    }
}

void FirstSpikeTimeNeuron::reset()
{
    // set time back to 0
    this->t = 0;
    // set spike delay back to none
    this->spike_delay = 0;
    // set dendrite back to 0
    this->dendrite = 0;
    // axon will not spike.
    this->axon = DelayedSpike();
}
