#include "base_snn.h"
#include "snn_ex.h"


BaseSNN::BaseSNN(unsigned int i_layer_size,
        std::vector<arma::Col<double>> d_init, double tau_m,
        double u_rest, double init_v, double t_reset, double k_nought,
        double round_zero, double alpha, double u_max)

{
    // run checks
    if(i_layer_size != d_init.size() || h_layer_size != d_init.at(0).size())
    {
        // invalid initial delays
        fprintf(stderr, "invalid initial delays {%u} - {%u}\n", i_layer_size, (unsigned int)d_init.size());
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
    if(DEBUG)
        printf("Passed all SRM constructor checks\n");
    
    // initialize variables
    this->i_layer_size = i_layer_size;
    this->h_layer_size = h_layer_size;
    this->t = 0;
    this->winner_neuron = NO_WINNER; // undefined value.
    this->d_ji = d_init;
    this->is_stimulus_new = true;

    
    // Populate layers with neurons
    for(unsigned int i = 0; i < i_layer_size; i++)
    {
        // create FSTN layer (input)
        this->input_layer.push_back(FirstSpikeTimeNeuron(i, alpha));
    }
    if(DEBUG)
        printf("Created FSTN layer\n");
    
    for(unsigned int h = 0; h < h_layer_size; h++)
    {
        // create hidden layer
        // d_init.at(0) is bologna. its is virtually useless!
        std::vector<double> tmp_d_init;
        for(unsigned int j = 0; j < i_layer_size; j++)
        {
            tmp_d_init.push_back(d_init.at(j).at(h));
        }
        /** Please edit after changing LFI model. **/
        this->hidden_layer.push_back(SpikeResponseModelNeuron(h, i_layer_size, 
        arma::Col<double>(tmp_d_init), tau_m, u_rest, init_v, t_reset,
        k_nought, round_zero, u_max));// 0, 0, arma::dvec() values are useless.
    }
    if(DEBUG)
        printf("Created Processing layer\n");
    // initialize network of spikes between input and hidden layers.
    queue_ji.resize(i_layer_size);
    for(unsigned int i = 0; i < i_layer_size; i++)
    {
        // for every input network connections
        // reserve a queue for every processing neuron
        queue_ji.at(i).resize(h_layer_size);
    }

    if(DEBUG)
    {
        printf("Initialized network\n");
    }

}

void BaseSNN::process()
{
    // make sure we have enough data to input.
    if(this->stimuli.size() != input_layer.size())
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
    for(unsigned int j = 0; j < i_layer_size; j++)
    {
        // Do we need to encode the stimulus?
        if(this->is_stimulus_new)
        {
            input_layer.at(j).dendrite = this->stimuli.at(j);
            input_layer.at(j).encode();
            // the flag should be turned off at the end of function
        }

        input_layer.at(j).t_pulse(); // we pulse after giving stimulus

        // if spikes are outputed, add delay to spike and place in queue
        // for each processing neuron in hiden layer
        if(input_layer.at(j).axon.signal)
        {
            // add delay and add to every hidden neuron
            for(unsigned int i = 0; i < h_layer_size; i++)
            {
                // we add +1 to delay to account for current time
                if(DEBUG)
                    printf("Added spike comming from input to queue of neuron %u\n", i);
                this->queue_ji.at(j).at(i).push_back(DelayedSpike(std::round(this->d_ji.at(j).at(i) + 1), true));
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
    for(unsigned int i = 0; i < h_layer_size; i++)
    {
        for(unsigned int j = 0; j < i_layer_size; j++)
        {
            // is the queue empty? No spikes?
            if(DEBUG)
                printf("Processing network {%u} -> {%u}\n", j, i);

            if(queue_ji.at(j).at(i).empty())
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
                // This flag is so that only 1 spike crosses the synaptic channel
                bool arrival_train_flag = false;
                for(unsigned int spike = 0; spike < queue_ji.at(j).at(i).size(); spike++)
                {
                    if(DEBUG)
                        printf("Moving time in synapse {%u} to {%u} spike number %u\n", j, i, spike);
                    // Note: It is biologically impossible for two spikes in the same synapse
                    // to be delivered at the same time. It is possible for the spikes
                    // from two or more input neurons to arrive at the same time, but
                    // this is accounted for by the design of the dendrites as a vector
                    // of spikes. The neurons create kappas for each spike!

                    // send spikes to dendrites of hidden layer
                    if(--queue_ji.at(j).at(i).at(spike).delay <= 0)
                    {
                        // Spike's delay is over
                        this->hidden_layer.at(i).dendrite.at(j) = DelayedSpike(0, true);
                        if(DEBUG)
                            printf("Spike arrived from input %u axon to processing dendrite number %u, spike: %d\n", j, i, this->hidden_layer.at(i).dendrite.at(j).signal);
                        // remove this spike. Has been delivered, not needed anymore.
                        this->queue_ji.at(j).at(i).erase(this->queue_ji.at(j).at(i).begin()+spike);

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
                }
                if(!arrival_train_flag) // if no spike arrived to ps neuron
                {
                    // spike's delay not over
                    this->hidden_layer.at(i).dendrite.at(j) = DelayedSpike();
                }
            }
        }

        // Pulse the ith neuron in the processing layer
        this->hidden_layer.at(i).t_pulse();

        // Was the neuron the first to spike (aka. won)?
        if(this->winner_neuron == NO_WINNER && this->hidden_layer.at(i).axon.signal)
        {
            this->winner_neuron = this->hidden_layer.at(i).snn_id;
        }
    }
    if(DEBUG)
        printf("Advanced the network's time\n");
   

    /**
     * The neural membrane will be affected until the next epoch?
     * Is it a way to have the feedback affect the neural membrane
     * in this epoch? like a limit? or something?!
    */

    this->is_stimulus_new = false;

}


void BaseSNN::reset()
{
    // reset time of network.
    this->t = 0;
    for(unsigned int neuron = 0; neuron < this->h_layer_size; neuron++)
    {
        // for every neuron, reset
        this->hidden_layer.at(neuron).reset();
    }
    for(unsigned int i = 0; i < this->i_layer_size; i++)
    {
        // for every input neuron, reset
        this->input_layer.at(i).reset();
    }
    this->winner_neuron = NO_WINNER;
    // reset delayed spikes on wait.
    for(unsigned int j = 0; j < this->queue_ji.size(); j++)
    {
        for(unsigned int i = 0; i < this->queue_ji.at(j).size(); i++)
        {
            this->queue_ji.at(j).at(i).clear();
        }
    }
}

void BaseSNN::re_process(std::vector<double> data)
{
    this->reset();
    this->stimuli = data;
    this->is_stimulus_new = true;
}