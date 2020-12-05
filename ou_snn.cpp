#include <math.h>
#include "ou_snn.h"


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
    this->t = delay;
}

bool OU_LIF_SRM::KappaFilter::is_zero()
{
    return this->near_zero;
}

double OU_LIF_SRM::KappaFilter::kappa(unsigned int s)
{
    return kappa_nought * exp( -s / this->tau_m);
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
        if(this->k < this->min_val)
        {
            this->near_zero = true;
        }
    }
}
/* End of OU_LIF_SRM::KappaFilter */


/* OU_LIF_SRM */




double OU_LIF_SRM::membrane_potential()
{
    // Sum of all membrane potentials from input layer at time t
    double k_sigma = 0;
    for(int i = 0; i < this->k_filter_list.size(); i++)
    {
        for(int k_i = 0; k_i < this->k_filter_list.at(i).size(); k_i++)
        {
            k_sigma += this->k_filter_list.at(i).at(k_i).k;
        }
    }
    // the sum of all kappas is the current membrane potential
    
    return k_sigma + this->u_rest;
}

OU_LIF_SRM::OU_LIF_SRM(unsigned int snn_id, arma::Col<double> init_w, 
double tau_m, double tau_s, double u_rest, double init_v, double n_inputs, 
unsigned char t_reset, double kappa_naugh, double min_val)
{   
    this->w_j = init_w;
    this->i = snn_id;
    this->tau_m = tau_m;
    this->tau_s = tau_s;
    this->u_rest = u_rest;
    this->v = init_v;
    this->t = 0;
    this->kappa_naugh = kappa_naugh;
    this->min_val = min_val;
    // fill in with empty k lists
    std::vector<KappaFilter> tmp;
    for(int input = 0; input < n_inputs; input++)
    {
        this->k_filter_list.push_back(tmp);
    }
    this->t_reset = t_reset;
    this->t_ref = -1;
    this->fired = false;
}

void OU_LIF_SRM::t_pulse()
{
    // march forward
    (this->t)++;
    // count to see if we reach the end of refactory period
    (this->t_ref)++;
    if(this->fired && this->t_ref == this->t_reset)
    {
        // ended refactory period
        this->fired = false;
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
        D_Spike out_spike(0, 0);
        this->axon = out_spike;
        // we go back
        // at this point all of the kappas that existed were removed.

    }
    // t_pulse all filters forward
    // also remove invalid near zero kappas
    for(int input = 0; i < k_filter_list.size(); i++)
    {
        for(int j = 0; j < k_filter_list.at(input).size(); j++)
        {
            k_filter_list.at(input).at(j).t_pulse();
            if(k_filter_list.at(input).at(j).is_zero())
            {
                // we remove this (jth) kappa. it is basically depleted
                k_filter_list.at(input).erase(k_filter_list.at(input).begin()+j);
            }
        }
    }
    // check the inputs from input layer to create kappas
    for (int input = 0; input < this->dendrite.size(); input++)
    {
        if(this->i == input)
        {
            continue;
        }
        if(this->dendrite.at(input).signal)
        {
            if(this->fired && this->dendrite.at(input).d < (this->t_reset - this->t_ref))
            {
                // we are in refactory period but this spike will arrive after it.
                KappaFilter new_spike(this->tau_m, this->kappa_naugh, this->min_val, dendrite.at(input).d);
                this->k_filter_list.at(input).push_back(new_spike);
            }
            else if(!this->fired)
            {
                // Spike arrived w/ delay. add new kappa.
                KappaFilter new_spike(this->tau_m, this->kappa_naugh, this->min_val, dendrite.at(input).d);
                this->k_filter_list.at(input).push_back(new_spike);
            }
            // if neuron is in refactory, ignore any spike with low delay
        }
    }
    // check for horizontal inputs?

    // solve the neuron
    if(!this->fired)
    {
        this->solve();
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
            for(int i = 0; i < this->k_filter_list.size(); i++)
            {
                this->k_filter_list.at(i).clear();
            }
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


