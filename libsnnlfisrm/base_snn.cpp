#include "base_snn.h"
#include "snn_ex.h"


BaseSNN::BaseSNN(unsigned int i_layer_size, unsigned int h_layer_size, 
        std::vector<arma::Col<double>> d_init, std::vector<arma::Col<double>> w_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, unsigned int n_x, unsigned int n_y, double neural_distance,
        double u_max)
{
    // run checks
    if(i_layer_size != d_init.size() || h_layer_size != d_init.at(0).size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%u} - {%u}\n", i_layer_size, (unsigned int)d_init.size());
        throw neuronexception();
    }
    if(h_layer_size != w_init.size() || h_layer_size != w_init.at(0).size())
    {
        // invalid initial weights
        fprintf(stderr, "invalid initial weights\n");
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
    if(k_nought <= 0)
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
    if(h_layer_size != n_y*n_x)
    {
        // invalid number of neurons and requested layout
        fprintf(stderr, "invalid number of neurons for requested layout.\n");
        throw neuronexception();
    }
    if(neural_distance <= 0)
    {
        // invalid neighboor distance
        fprintf(stderr, "invalid neighboor neuron distance\n");
        throw neuronexception();
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
        this->input_layer.push_back(FirstSpikeTimeNeuron(i, alpha));
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

        this->hidden_layer.push_back(SpikeResponseModelNeuron(h, i_layer_size, 
        h_layer_size, arma::Col<double>(tmp_d_init), w_init.at(h), tau_m, u_rest, init_v, t_reset,
        k_nought, round_zero, x, y, u_max));
        if(++x >= n_x)
        {
            x = 0;
            y++;
        }
        if(y > n_y)
        {
            fprintf(stderr, "Error!\n");
            throw neuronexception();
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
    {
        printf("Initialized network\n");
    }
        this->has_winner = false;
        this->winner_neuron = -1; // undefined value.
}

void BaseSNN::process(std::vector<double> data)
{
    // make sure we have enough data to input.
    if(data.size() != input_layer.size())
    {
        // no enough data
        fprintf(stderr, "No enough data passed\n");
        throw InputLayerException();

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
                this->net_queue_i.at(j).at(i).push_back(DelayedSpike(this->d_ji.at(j).at(i) + 1, true));
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
                this->hidden_layer.at(i).dendrite.at(j) = DelayedSpike();
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
                    if(--net_queue_i.at(j).at(i).at(spike).delay <= 0)
                    {
                        // Spike's delay is over
                        this->hidden_layer.at(i).dendrite.at(j) = DelayedSpike(0, true);
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
                        this->hidden_layer.at(i).dendrite.at(j) = DelayedSpike();
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
    std::vector<DelayedSpike> tmp_queue_m;
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

unsigned int BaseSNN::find_winner_spike()
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

void BaseSNN::reset()
{
    // reset time of network.
    this->t = 0;
    for(unsigned int neuron = 0; neuron < this->h_size; neuron++)
    {
        // for every neuron, reset
        this->hidden_layer.at(neuron).reset();
    }
    for(unsigned int i = 0; i < this->i_size; i++)
    {
        // for every input neuron, reset
        this->input_layer.at(i).reset();
    }
    this->winner_neuron = -1;
    this->has_winner = false;
    // reset delayed spikes on wait.
    for(unsigned int j = 0; j < this->net_queue_i.size(); j++)
    {
        for(unsigned int i = 0; i < this->net_queue_i.at(j).size(); i++)
        {
            this->net_queue_i.at(j).at(i).clear();
        }
    }
    // reset lateral synapses
    net_queue_m.clear();
    net_queue_m.resize(this->h_size);

}
