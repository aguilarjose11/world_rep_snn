#include <math.h>
#include <exception>
#include <algorithm>
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

class euclideanexception: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Invalid euclidean inputs";
  }
} eucex;


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
        double kappa_naugh, double round_zero, double x, double y)
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
    this->x = x;
    this->y = y;
    this->disabled = false;

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
                    printf("Spike Recieved. Adding kappa.\n");
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
    if(!this->fired && !this->disabled)
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
    else
    {
        // no firing
        D_Spike no_fire_spike(0, 0);
        this->axon = no_fire_spike;
    }
    
    
}

bool OU_LIF_SRM::fire_condition(double u)
{
    return u >= this->v;
}

void OU_LIF_SRM::disable()
{
    this->disabled = true;
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
    if(DEBUG)
        printf("Delay to be calculated\n");
    this->spike_delay = (unsigned int) abs(this->alpha * this->dendrite);
    // check requirements to send action potential
    if((this->spike_delay == 0) || this->t % this->spike_delay == 0)
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
        std::vector<arma::Col<double>> d_init, std::vector<arma::Col<double>> w_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, unsigned int n_x, unsigned int n_y, double neural_distance)
{
    // run checks
    if(i_layer_size != d_init.size() || h_layer_size != d_init.at(0).size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%u} - {%u}\n", i_layer_size, (unsigned int)d_init.size());
        throw neuronex;
    }
    if(h_layer_size != w_init.size() || h_layer_size != w_init.at(0).size())
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
    if(h_layer_size != n_y*n_x)
    {
        // invalid number of neurons and requested layout
        fprintf(stderr, "invalid number of neurons for requested layout.\n");
        throw neuronex;
    }
    if(neural_distance <= 0)
    {
        // invalid neighboor distance
        fprintf(stderr, "invalid neighboor neuron distance\n");
        throw neuronex;
    }
    if(DEBUG)
        printf("Passed all SRM constructor checks\n");
    
    // initialize variable
    this->t = 0;
    this->i_size = i_layer_size;
    this->h_size = h_layer_size;
    this->d_ji = d_init;
    this->n_x = n_x;
    this->n_y = n_y;
    // neighboor distance. preferably 1
    this->neural_distance = neural_distance;
    
    // Populate layers with neurons
    for(unsigned int i = 0; i < i_layer_size; i++)
    {
        // create FSTN layer (input)
        this->input_layer.push_back(OU_FSTN(i, alpha));
    }
    if(DEBUG)
        printf("Created FSTN layer\n");
    
    for(unsigned int h = 0, x = 0, y = 0; h < h_layer_size; h++)
    {
        // create hidden layer
        // d_init.at(0) is bologna. its is virtually useless!
        std::vector<double> tmp_d_init;
        for(unsigned int j = 0; j < i_layer_size; j++)
        {
            tmp_d_init.push_back(d_init.at(j).at(h));
        }

        this->hidden_layer.push_back(OU_LIF_SRM(h, i_layer_size, 
        h_layer_size, arma::Col<double>(tmp_d_init), w_init.at(h), tau_m, u_rest, init_v, t_reset,
        k_nought, round_zero, x, y));
        if(++x >= n_x)
        {
            x = 0;
            y++;
        }
        if(y > n_y)
        {
            fprintf(stderr, "Error!\n");
            throw neuronex;
        }
    }
    if(DEBUG)
        printf("Created Processing layer\n");
    // initialize network of spikes between input and hidden layers.
    net_queue_i.resize(i_layer_size);
    for(unsigned int i = 0; i < i_layer_size; i++)
    {
        // for every input network connections
        // reserve a queue for every processing neuron
        net_queue_i.at(i).resize(h_layer_size);
    }

    // initialize network of spikes between neurons in hidden layer
    net_queue_m.resize(h_layer_size);
    if(DEBUG)
        printf("Initialized network\n");
}

void OU_SRM_NET::process(std::vector<double> data)
{
    // make sure we have enough data to input.
    if(data.size() != input_layer.size())
    {
        // no enough data
        fprintf(stderr, "No enough data passed\n");
        throw ilex;

    }

    if(DEBUG)
        printf("Passed all process checks\n");

    // March time forward
    (this->t)++;

    /**
     * - Put data into dendrites of coding neurons.
     * - Pulse the neuron to do processing.
     * - Grab the axon output, delay, and add to the
     *   individual queue of each processing neuron 
     *   in hidden layer
    */
    for(unsigned int j = 0; j < i_size; j++)
    {
        input_layer.at(j).dendrite = data.at(j);
        input_layer.at(j).t_pulse();
        // if spikes are outputed, add delay to spike and place in queue
        // for each processing neuron in hiden layer
        if(input_layer.at(j).axon.signal)
        {
            // add delay and add to every hidden neuron
            for(unsigned int i = 0; i < h_size; i++)
            {
                // we add +1 to delay to account for current time
                if(DEBUG)
                    printf("Added spike comming from input to queue of neuron %u\n", i);
                this->net_queue_i.at(j).at(i).push_back(D_Spike(this->d_ji.at(j).at(i) + 1, true));
            }
        }
    }
    if(DEBUG)
        printf("Added data into dendrites of input neurons\n");

    /**
     * Advance time in network connecting the input layer and the hidden
     * layers.
     * 
     * - Reduce the left delay time for each neuron's queue
     * - Send the active or innactive spikes to the corresponding hidden
     *   layer's neuron's dendrites.
    */
    for(unsigned int j = 0; j < i_size; j++)
    {
        for(unsigned int i = 0; i < h_size; i++)
        {
            // is the queue empty? No spikes?
            if(DEBUG)
                printf("Processing network {%u} -> {%u}\n", j, i);

            if(net_queue_i.at(j).at(i).empty())
            {
                // send inactive spikes
                if(DEBUG)
                    printf("Filling dendrite\n");
                this->hidden_layer.at(i).dendrite.at(j) = D_Spike();
                if(DEBUG)
                    printf("Empty spike train queue from input {%u} to {%u}\n", j, i);
            }
            else
            {
                // We have spikes to process!
                bool arrival_train_flag = false;
                for(unsigned int spike = 0; spike < net_queue_i.at(j).at(i).size(); spike++)
                {
                    if(DEBUG)
                        printf("Moving time in synapse {%u} to {%u} spike number %u\n", j, i, spike);
                    // Note: It is biologically impossible for two spikes in the same synapse
                    // to be delivered at the same time. It is possible for the spikes
                    // from two or more input neurons to arrive at the same time, but
                    // this is accounted for by the design of the dendrites as a vector
                    // of spikes. The neurons create kappas for each spike!

                    // send spikes to dendrites of hidden layer
                    if(--net_queue_i.at(j).at(i).at(spike).d <= 0)
                    {
                        // Spike's delay is over
                        this->hidden_layer.at(i).dendrite.at(j) = D_Spike(0, true);
                        if(DEBUG)
                            printf("Spike arrived from input %u axon to processing dendrite number %u, spike: %d\n", j, i, this->hidden_layer.at(i).dendrite.at(j).signal);
                        // remove this spike. Has been delivered, not needed anymore.
                        this->net_queue_i.at(j).at(i).erase(this->net_queue_i.at(j).at(i).begin()+spike);

                        // account for the shift in vector after deletion.
                        spike--;

                        /**
                         * Once a spike arrives, no more spikes can arrive,
                         * 
                         * I can avoid the use of flags by setting the dendrite back to nothing and only add
                         * a positive spike when it happens!
                        */
                        arrival_train_flag = true;
                        
                    }
                    else if(!arrival_train_flag)
                    {
                        // spike's delay not over
                        this->hidden_layer.at(i).dendrite.at(j) = D_Spike();
                    }
                }
            }
        }
    }
    if(DEBUG)
        printf("Advanced the network's time\n");
    /**
     * Advance time in lateral feedback network in hidden layer
     * 
     * Note for myself:
     * We can add the result of feedback back into the neuron
     * by recalculating the membrane potential
     * 
     * At the same time, it could be a bit more accurate keeping
     * a delay of exactly 1 t?
    */

    // Create queue network for next epoch.
    std::vector<D_Spike> tmp_queue_m;
    tmp_queue_m.resize(this->h_size);

    for(unsigned int i = 0; i < this->hidden_layer.size(); i++)
    {
        // plug in the  lateral spikes into neuron
        this->hidden_layer.at(i).horizontal_dendrite = this->net_queue_m;

        // advance time in hidden layer/process
        if(DEBUG)
            printf("dendritic inputs for neuron %u: %d, %d\n", i, this->hidden_layer.at(i).dendrite.at(0).signal, this->hidden_layer.at(i).dendrite.at(1).signal);
        this->hidden_layer.at(i).t_pulse();

        // catch the hiden layer's axon and put it in feedback network
        tmp_queue_m.at(i) = this->hidden_layer.at(i).axon;
    }
    if(DEBUG)
        printf("Advanced/executed hidden layer neurons\n");

    // Update the spike train for lateral synapses.
    this->net_queue_m = tmp_queue_m;

    /**
     * The neural membrane will be affected until the next epoch?
     * Is it a way to have the feedback affect the neural membrane
     * in this epoch? like a limit? or something?!
    */
}

unsigned int OU_SRM_NET::find_winner_spike()
{
    if(this->has_winner)
    {
        return this->winner_neuron;
    }
    else
    {
        for(unsigned int i = 0; i < this->h_size; i++)
        {
            if(this->hidden_layer.at(i).fired)
            {
                // we found a spike that just fired!
                this->winner_neuron = i;
                this->has_winner = true;
                return this->winner_neuron;
            }
        }
        return NO_WINNER_SPIKE;
    }
    
}


/* Distance Matrix builder */

arma::Mat<double> euclidean_distance_matrix(OU_SRM_NET *snn_net, double distance_unit)
{
    arma::Mat<double> distance_matrix(snn_net->h_size, snn_net->h_size);
    double euclidean_dist = 0.0;
    double x_1, x_2, y_1, y_2;
    if(DEBUG)
        printf("Matrix Size: %u x %u\n", snn_net->h_size, snn_net->h_size);
    
    for(unsigned int neuron_m = 0; neuron_m < snn_net->h_size; neuron_m++)
    {
        for(unsigned int neuron_n = 0; neuron_n < snn_net->h_size; neuron_n++)
        {
            x_1 = snn_net->hidden_layer.at(neuron_m).x;
            y_1 = snn_net->hidden_layer.at(neuron_m).y;
            x_2 = snn_net->hidden_layer.at(neuron_n).x;
            y_2 = snn_net->hidden_layer.at(neuron_n).y;
            euclidean_dist = distance_unit * sqrt(pow(x_2 - x_1, 2) + pow(y_2 - y_1, 2));
            if(DEBUG)
                printf("Euclidean calculation: %f\n", euclidean_dist);
            distance_matrix(neuron_m, neuron_n) = euclidean_dist;
        }
        if(DEBUG)
            printf("neuron %u -> (%f, %f)\n", neuron_m, x_1, y_1);
    }
    return distance_matrix;
}

arma::Mat<double> euclidean_distance_matrix(std::vector<std::vector<double>> *point_list, double distance_unit)
{
    const unsigned int X_INDEX = 0, Y_INDEX = 1;
    if(point_list->at(X_INDEX).size() != point_list->at(Y_INDEX).size())
    {
        // Problem with the input points
        fprintf(stderr, "Input Point matrix is not well shapped.\n");
        throw eucex;
    }
    arma::Mat<double> distance_matrix(point_list->at(X_INDEX).size(), point_list->at(0).size());
    double euclidean_dist = 0.0;
    double x_1, x_2, y_1, y_2;
    if(DEBUG)
        printf("Matrix Size: %u x %u\n", (unsigned int)point_list->at(0).size(), (unsigned int)point_list->at(0).size());
    
    for(unsigned int point_m = 0; point_m < point_list->at(0).size(); point_m++)
    {
        for(unsigned int point_n = 0; point_n < point_list->at(0).size(); point_n++)
        {
            x_1 = point_list->at(X_INDEX).at(point_m);
            y_1 = point_list->at(Y_INDEX).at(point_m);
            x_2 = point_list->at(X_INDEX).at(point_n);
            y_2 = point_list->at(Y_INDEX).at(point_n);
            euclidean_dist = distance_unit * sqrt(pow(x_2 - x_1, 2) + pow(y_2 - y_1, 2));
            if(DEBUG)
                printf("Euclidean calculation: %f\n", euclidean_dist);
            distance_matrix(point_m, point_n) = euclidean_dist;
        }
        if(DEBUG)
            printf("neuron %u -> (%f, %f)\n", point_m, x_1, y_1);
    }
    return distance_matrix;
}

/* Initial weight calculator */

std::vector<arma::Col<double>> initial_weight_euclidean(arma::Mat<double> distance_matrix, double sigma_1, double sigma_2)
{
    // Calculate initial weight formula
    arma::Mat<double> weight_matrix = 2*arma::exp(-arma::pow(distance_matrix, 2)/2*pow(sigma_1, 2)) - arma::exp(-arma::pow(distance_matrix, 2)/2*pow(sigma_2, 2));
    if(DEBUG)
    {
        printf("Matrix Content before weights\n\n");
        for(unsigned int i = 0; i < weight_matrix.n_rows; i++)
        {
            for(unsigned int j = 0; j < weight_matrix.n_cols; j++)
            {
                printf("%f\t", weight_matrix(i, j));
            }
            printf("\n");
        }
    }
    // Transpose to simplify data extraction
    weight_matrix = arma::trans(weight_matrix);
    // Create final data structure
    std::vector<arma::Col<double>> weight_struct;
    // fill final data structure
    for(unsigned int i = 0; i < weight_matrix.n_rows; i++)
    {
        weight_struct.push_back(arma::Col<double>(weight_matrix.col(i)));
    }
    return weight_struct;
}

std::vector<arma::Col<double>> initial_delay_vectors(unsigned int n_neurons, unsigned int n_delays, double l_bound, double h_bound)
{
    // run checks to input 
    if(l_bound > h_bound)
    {
        // invalid lower and upper ranges
        fprintf(stderr, "Invalid range for initial delay vector");
        throw eucex;
    }
    if(DEBUG)
        printf("Passed initial delay vector's checks.\n");
    // Initialize pseudo-random number generator
    srand(time(NULL));
    // Calculate range's distance
    double range = h_bound - l_bound;
    // Create and initialize delay matrix
    std::vector<arma::Col<double>> delay_matrix(n_neurons, arma::Col<double>(n_delays));
    // fill in random delay matrix
    for(unsigned int i = 0; i < n_neurons; i++)
    {
        for(unsigned int j = 0; j < n_delays; j++)
        {
            delay_matrix.at(i).at(j) = l_bound + (((double)rand()/RAND_MAX) * range);
        }
    }
    return delay_matrix;
}


std::vector<std::vector<double>> euclidean_point_map(unsigned int x, unsigned int y)
{
    std::vector<std::vector<double>> map_(2, std::vector<double>(x*y));
    for(unsigned int n = 0, n_point = 0; n < x; n++)
    {
        for(unsigned int m = 0; m < y; m++, n_point++)
        {
            map_.at(0).at(n_point) = n;
            map_.at(1).at(n_point) = m;
        }
    }
    return map_;
}

OU_SRMN_TRAIN::OU_SRMN_TRAIN(unsigned int i_layer_size, unsigned int h_layer_size, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, unsigned int n_x, unsigned int n_y, double neural_distance, 
        double distance_unit, double sigma_1, double sigma_2, double l_bound, double u_bound,
        double sigma_neighbor, double tau_alpha, double tau_beta, double eta_w, double eta_d,
        unsigned int t_max, unsigned int t_delta, double ltd_max)
{
    // No running checks here!
    // Set up network variables

    // Set up Neural Network

    std::vector<arma::Col<double>> d_init = initial_delay_vectors(i_layer_size, h_layer_size, l_bound, u_bound);
    // create distance matrix between nodes. This will be spatial in nature
    this->point_map = euclidean_point_map(n_x, n_y);
    // Create spatial distance matrix. Not really a lot of information unless combined with delays.
    this->distance_matrix = euclidean_distance_matrix(&(this->point_map), distance_unit);
    // use distance matrix to create initial weight matrix
    std::vector<arma::Col<double>> w_init = initial_weight_euclidean(distance_matrix, sigma_1, sigma_2);
    // initialize the layer.
    this->snn = new OU_SRM_NET(i_layer_size, h_layer_size, d_init, w_init, tau_m, u_rest, init_v, t_reset, k_nought, round_zero, alpha, n_x, n_y, neural_distance);

    // store hyperparameters
    this->sigma_neighbor = sigma_neighbor;
    this->tau_alpha = tau_alpha;
    this->tau_beta = tau_beta;
    this->eta_w = eta_w;
    this->eta_d = eta_d;
    this->t_max = t_max;
    this->t_delta = t_delta;
    this->ltd_max = ltd_max;
}

void OU_SRMN_TRAIN::train(std::vector<std::vector<double>> X)
{
    // training samples
    std::vector<double> sample = X.at(0);
    // Has the algorith timed out?
    bool time_out = false;
    // for every training sample...
    for(unsigned int n_sample = 0; n_sample < X.size(); n_sample++, sample = X.at(n_sample))
    {
        // feed training sample and run algorithm until a spike is found
        time_out = false;
        for(unsigned int t_t = 0; t_t < this->t_max; t_t++)
        {
            // process the current point until we have a spike
            this->snn->process(sample);
            // do we have a winner spike?
            

            if(this->snn->find_winner_spike() != NO_WINNER_SPIKE)
            {
                // update weights
                // make shallow copy of SNN
                OU_SRM_NET snn_copy = *(this->snn);
                // disable the winner neuron
                snn_copy.hidden_layer.at(this->snn->winner_neuron).disable();
                // Add winner neuron to list of fired neurons
                std::vector<unsigned int> fired_neurons;
                fired_neurons.push_back(this->snn->winner_neuron);
                // pass to weight updater to find possible neurons triggered by winner
                this->update_weights(snn_copy, &fired_neurons, this->t_delta, sample, this->snn->winner_neuron);
                // excitatory weights should have been updated by now.
                // update inhibitory 
                this->update_inhibitory_weights(&fired_neurons);
                // modify delays
                update_delays(sample);
                // reset neuron with new delays/weights and input neurons
                this->snn->reset();
                // break out...
                break;
            }
        }
        // have we timed out?
        if(time_out)
        {
            // yep. No change in weights ot delays.
            fprintf(stderr, "WARNING (Timeout): No winner spike exists after %u epochs. Maybe increase t_max or change the other hyperparameters?\n", this->t_max);
        }
    }
    // At this point we should have finished our training.
    // consider setting everything back to zero if needed to retrain
}

double OU_SRMN_TRAIN::neighbor(OU_LIF_SRM m, OU_LIF_SRM n)
{
    // return general neighbor function
    return exp(pow(this->distance_matrix(m.snn_id, n.snn_id), 2) / (2*pow(this->sigma_neighbor, 2)));
}

void OU_SRMN_TRAIN::update_delays(std::vector<double> sample)
{
    // for every input neuron
    for(unsigned int j = 0; j < this->snn->i_size; j++)
    {
        // for every processing neuron
        for(unsigned int m = 0; m < this->snn->h_size; m++)
        {
            double delta_d = this->eta_d*this->neighbor(this->snn->hidden_layer.at(m), this->snn->hidden_layer.at(this->snn->winner_neuron))*(sample.at(j)-this->snn->d_ji.at(j).at(m));
            this->snn->d_ji.at(j).at(m) += delta_d;
        }
    }
}

/**
 * Note that when an excitatory connection happens, also, an inhibitory connection sould be placed between the postsynaptic and presynaptic neurons:
 * So we place LTP the the connection M->N, but we utilize a LTD for M<-N.
 * This is because if we see neuron N from the perspective of it being a presynaptic neuron, we will realize that t_n>t_m. because of this, we shall
 * apply LTD. this practically "distances" the neurons and only connects the ones that may have things in common.
 * Even more important to keep track of the connections we modify!
*/
void OU_SRMN_TRAIN::update_weights(OU_SRM_NET network, std::vector<unsigned int> *fired_neurons, unsigned int t_epoch, std::vector<double> sample, unsigned int fired_neuron)
{
    // Note: maybe do epoch + 1 for updating function?
    for(unsigned int epoch = 0; epoch < t_epoch; epoch++)
    {
        // run t_epoch times to see if we find a triggered spike
        network.process(sample);
        for(unsigned int neuron = 0; neuron < network.h_size; neuron++)
        {  
            // Has a neuron fired?
            if(network.hidden_layer.at(neuron).axon.signal)
            {
                //yep
                // update weights
                // excitatory change
                double delta_w_ij = this->eta_w*neighbor(network.hidden_layer.at(fired_neuron), network.hidden_layer.at(neuron))*(1-this->snn->hidden_layer.at(neuron).w_m.at(fired_neuron))*exp(epoch/this->tau_alpha);
                this->snn->hidden_layer.at(neuron).w_m.at(fired_neuron) += delta_w_ij;
                // inhibitory change
                double delta_w_ji = -this->eta_w*neighbor(network.hidden_layer.at(neuron), network.hidden_layer.at(fired_neuron))*(1+this->snn->hidden_layer.at(fired_neuron).w_m.at(neuron))*exp(epoch/this->tau_beta);
                this->snn->hidden_layer.at(fired_neuron).w_m.at(neuron) += delta_w_ji;
                // add neuron to list of fired neurons
                fired_neurons->push_back(neuron);
                // make a copy of the network
                OU_SRM_NET n_net = network;
                // disable fired neuron
                network.hidden_layer.at(neuron).disable();
                // update weights recursively
                update_weights(n_net, fired_neurons, t_epoch, sample, neuron);
                // by the time that ends, we continue on our analysis.
                continue;


            }
        }
    }
    // by this point all neurons should have been updated!
}

/**
 * Note that we are using a constant value for the amount of change. This may not affect excitatory synapses,
 * but for inhibitory it just may.
*/
void OU_SRMN_TRAIN::update_inhibitory_weights(std::vector<unsigned int> *updated_synapses)
{
    // for every neuron...
    for(unsigned int m = 0; m < this->snn->h_size; m++)
    {
        // for every neuron ahead of this one (m)
        for(unsigned int n = m + 1; n < this->snn->h_size; m++)
        {
            // check if the synapses were modified previously.
            if(std::find(updated_synapses->begin(), updated_synapses->end(), m) != updated_synapses->end() && 
               std::find(updated_synapses->begin(), updated_synapses->end(), n) != updated_synapses->end())
            {
                // These neurons were updated. we skip them!
                continue;
            }
            // update the synapses as inhibitory
            // Testing values: using t_max. possible: t_delta.
            double delta_w_mn = -this->eta_w*neighbor(this->snn->hidden_layer.at(m), this->snn->hidden_layer.at(n))*(1+this->snn->hidden_layer.at(n).w_m.at(m))*exp(this->t_max/this->tau_beta);
            this->snn->hidden_layer.at(n).w_m.at(m) += delta_w_mn;
            double delta_w_nm = -this->eta_w*neighbor(this->snn->hidden_layer.at(n), this->snn->hidden_layer.at(m))*(1+this->snn->hidden_layer.at(m).w_m.at(n))*exp(this->t_max/this->tau_beta);
            this->snn->hidden_layer.at(m).w_m.at(n) += delta_w_nm;
            // values updated. anything else? nah.
        }
    }
    // inhibitory connections updated!
}