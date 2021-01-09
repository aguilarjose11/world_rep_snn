#include <armadillo>
#include <vector>

#include "lfi.h"
#include "base_snn.h"
#include "snn.h"


neuronexception neuronex = neuronexception();

InputLayerException ilex = InputLayerException();

euclideanexception eucex = euclideanexception();

/* Distance Matrix builder */

arma::Mat<double> euclidean_distance_matrix(BaseSNN *snn_net, double distance_unit)
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

SNN::SNN(unsigned int i_layer_size, unsigned int h_layer_size, double tau_m,
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
    this->snn = new BaseSNN(i_layer_size, h_layer_size, d_init, w_init, tau_m, u_rest, init_v, t_reset, k_nought, round_zero, alpha, n_x, n_y, neural_distance);
    if(DEBUG)
        printf("Passed network initializarion.\n");
    // store hyperparameters
    this->sigma_neighbor = sigma_neighbor;
    this->tau_alpha = tau_alpha;
    this->tau_beta = tau_beta;
    this->eta_w = eta_w;
    this->eta_d = eta_d;
    this->t_max = t_max;
    this->t_delta = t_delta;
    this->ltd_max = ltd_max;
    if(DEBUG)
        printf("Completed initialization.\n");
}

void SNN::train(std::vector<std::vector<double>> X)
{
    // Create constants
    const unsigned int X_DATA = 0;
    const unsigned int Y_DATA = 1;
    const unsigned int UNDEFINED_TIME = (UINT_MAX) - 2;
    // sample data
    std::vector<double> sample;
    // initialize the sample data
    sample.resize(2);
    // Not really necesary since we do this inside loop
    sample.at(X_DATA) = X.at(X_DATA).at(0);
    sample.at(Y_DATA) = X.at(Y_DATA).at(0);
    // timeout flag
    bool time_out = true;
    // Time table per sample. Useful to calculate weights
    std::vector<unsigned int> time_table;
    // for every training sample...
    for(unsigned int n_sample = 0; n_sample < X.at(X_DATA).size(); n_sample++)
    {
        // Set time out flag ready
        time_out = true;
        // set up the current sample from the X dataset
        sample.at(X_DATA) = X.at(X_DATA).at(n_sample);
        sample.at(Y_DATA) = X.at(Y_DATA).at(n_sample);
        /**
         * We clear and resize the timetable to contain the times
         * for every single processing neuron.
         * 
         * This table will help us find the weight change between
         * every single neuron
        */
        time_table.clear();
        time_table.resize(this->snn->h_size, UNDEFINED_TIME);
        
        // We will run the same data up to t_max times
        // if no spike is generated, we say that we timed out
        for(unsigned int t_t = 0; t_t < this->t_max; t_t++)
        {
            // process the current point until we have a spike
            this->snn->process(sample);
            // Is there any spikes?
            for(unsigned int m = 0; m < this->snn->h_size; m++)
            {
                // look for any neurons that may have spiked
                if(time_table.at(m) == UNDEFINED_TIME && this->snn->hidden_layer.at(m).axon.signal)
                {
                    /**
                     * We add the time of spike to time table.
                     * We only keep the earliest time of spike
                     * on the time table. It is totally possible for
                     * the spike to spike a second time but not kept
                    */
                   if(!(this->snn->has_winner))
                   {
                       // we look for a winner neuron in the system.
                       this->snn->find_winner_spike();
                   }
                   time_table.at(m) = t_t;
                   if(time_out)
                   {
                       // we set the time out flag to false. We found at least one spike
                       time_out = false;
                   }
                }
            }
        }
        // have we timed out?
        if(time_out)
        {
            // yep. No change in weights ot delays.
            fprintf(stderr, "WARNING (Timeout): No winner spike exists after %u epochs. Maybe increase t_max or change the other hyperparameters?\n", this->t_max);
            // reset neuron with new delays/weights and input neurons
            this->snn->reset();
        }
        else // We got a winner neuron!
        {
            /**
             * For any times that are UNDEFINED_TIME, it is assumed that the spike time for these neurons is so
             * far in the future that we can treat any changes to the weights as basically minuscule.
            */
            // Update weights
            for(unsigned int m = 0; m < this->snn->h_size; m++)
            {
                // for every neuron ahead of this one.
                for(unsigned int n = m + 1; n < this->snn->h_size; n++)
                {
                    unsigned int t_m = time_table.at(m);
                    unsigned int t_n = time_table.at(n);
                    if((t_m != UNDEFINED_TIME || t_n != UNDEFINED_TIME) && t_m >= t_n)
                    {
                        // excitatory m->n
                        double delta_w_mn = this->eta_w*neighbor(this->snn->hidden_layer.at(m), this->snn->hidden_layer.at(n))*(1-this->snn->hidden_layer.at(n).w_m.at(m))*std::exp((t_m-t_n)/this->tau_alpha);
                        this->snn->hidden_layer.at(n).w_m.at(m) += delta_w_mn;

                        // inhibitory n->m
                        double delta_w_nm = this->eta_w*neighbor(this->snn->hidden_layer.at(n), this->snn->hidden_layer.at(m))*(1+this->snn->hidden_layer.at(m).w_m.at(n))*std::exp((t_m-t_n)/this->tau_beta);
                        this->snn->hidden_layer.at(m).w_m.at(n) -= delta_w_nm;
                        if(DEBUG)
                        {
                            printf("%f %f\n", std::exp(t_m-t_n/this->tau_alpha), std::exp(t_m-t_n/this->tau_beta));
                            printf("%f %f\n", delta_w_mn, delta_w_nm);
                        }
                    }
                    else if((t_m != UNDEFINED_TIME || t_n != UNDEFINED_TIME) && t_m < t_n)
                    {
                        // inhibitory m->n
                        double delta_w_mn = this->eta_w*neighbor(this->snn->hidden_layer.at(m), this->snn->hidden_layer.at(n))*(1+this->snn->hidden_layer.at(n).w_m.at(m))*std::exp((t_n-t_m)/this->tau_beta);
                        this->snn->hidden_layer.at(n).w_m.at(m) -= delta_w_mn;
                        // excitatory n->m
                        double delta_w_nm = this->eta_w*neighbor(this->snn->hidden_layer.at(n), this->snn->hidden_layer.at(m))*(1-this->snn->hidden_layer.at(m).w_m.at(n))*std::exp((t_n-t_m)/this->tau_alpha);
                        this->snn->hidden_layer.at(m).w_m.at(n) += delta_w_nm;
                        if(DEBUG)
                            printf("%f %f\n", delta_w_mn, delta_w_nm);
                    }
                    else
                    {
                        if(DEBUG);
                    }
                }
            }

            // there is a winner neuron. Update delays
            for(unsigned int j = 0; j < this->snn->i_size; j++)
            {
                // for every input neuron
                printf("[ ");
                for(unsigned int m = 0; m < this->snn->h_size; m++)
                {
                    // for every processing neuron
                    // I know, i do not use 'this'. It is a comple 
                    double delta_mj = eta_d*neighbor(snn->hidden_layer.at(m), snn->hidden_layer.at(this->snn->winner_neuron))*(sample.at(j) - snn->d_ji.at(j).at(m));
                    this->snn->d_ji.at(j).at(m) += delta_mj;
                    printf(" %f ", delta_mj);
                }
                printf("]\n");
                //std::getchar();
            }
        }
        // reset network
        this->snn->reset();
        
        
    }
    // At this point we should have finished our training.
    // consider setting everything back to zero if needed to retrain
}

double SNN::neighbor(SpikeResponseModelNeuron m, SpikeResponseModelNeuron n)
{
    // return general neighbor function
    if(DEBUG)
        printf("e^(-%f^2/2*%f^2): %f\n", this->distance_matrix(m.snn_id, n.snn_id), std::pow(this->sigma_neighbor, 2), std::exp(-std::pow(this->distance_matrix(m.snn_id, n.snn_id), 2) / (2*std::pow(this->sigma_neighbor, 2))));
    return std::exp(-std::pow(this->distance_matrix(m.snn_id, n.snn_id), 2) / (2*std::pow(this->sigma_neighbor, 2)));
}

void SNN::update_delays(std::vector<double> sample)
{
    // for every input neuron
    for(unsigned int j = 0; j < this->snn->i_size; j++)
    {
        // for every processing neuron's synapse
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
void SNN::update_weights(BaseSNN network, std::vector<unsigned int> *fired_neurons, unsigned int t_epoch, std::vector<double> sample, unsigned int fired_neuron)
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
                BaseSNN n_net = network;
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
void SNN::update_inhibitory_weights(std::vector<unsigned int> *updated_synapses)
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