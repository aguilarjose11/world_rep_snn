#include "snn.h"


/* Distance Matrix builder */

arma::Mat<double> euclidean_distance_matrix(BaseSNN *snn_net, double distance_unit)
{
    arma::Mat<double> distance_matrix(snn_net->h_layer_size, snn_net->h_layer_size);
    double euclidean_dist = 0.0;
    double x_1, x_2, y_1, y_2;
    if(DEBUG)
        printf("Matrix Size: %u x %u\n", snn_net->h_layer_size, snn_net->h_layer_size);
    
    for(unsigned int neuron_m = 0; neuron_m < snn_net->h_layer_size; neuron_m++)
    {
        for(unsigned int neuron_n = 0; neuron_n < snn_net->h_layer_size; neuron_n++)
        {
            x_1 = snn_net->hidden_layer.at(neuron_m).d_j.at(0);
            y_1 = snn_net->hidden_layer.at(neuron_m).d_j.at(1); 
            x_2 = snn_net->hidden_layer.at(neuron_n).d_j.at(0);
            y_2 = snn_net->hidden_layer.at(neuron_n).d_j.at(1);
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
        throw euclideanexception();
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

std::vector<std::vector<double>> initial_delay_vector_2d_map(unsigned int n_x, 
unsigned int n_y, unsigned int delays_per_row, unsigned int delays_per_column)
{
    
    double x_step = (double)n_x / (double)delays_per_row;
    double y_step = (double)n_y / (double)delays_per_column;
    if(DEBUG)
        printf("%u %u %u %u, %f, %f\n", n_x, n_y, delays_per_row, delays_per_column, x_step, y_step);

    if(DEBUG)
        printf("creating delay matrix\n");
    // column 0 -> x, 1 -> y
    std::vector<std::vector<double>> delay_matrix;
    delay_matrix.resize(2);
    delay_matrix.at(0).resize(delays_per_row * delays_per_column);
    delay_matrix.at(1).resize(delays_per_row * delays_per_column);
    if(DEBUG)
        printf("Initialized delay matrix\n");
    unsigned int curr_delay = 0;
    for(double y = 0; y < n_y; y += y_step)
    {
        for(double x = 0; x < n_x; x += x_step)
        {
            delay_matrix.at(0).at(curr_delay) = x;
            delay_matrix.at(1).at(curr_delay) = y;
            if(DEBUG)
                printf("temporal flag %u\n", curr_delay);
            curr_delay++;
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

SNN::SNN(unsigned int n_data, double tau_m, double u_rest, double init_v, 
        double t_reset, double k_nought, double round_zero, double alpha, 
        unsigned int n_x, unsigned int n_y, double delay_distance, 
        double distance_unit, double sigma_neighbor, double eta_d,
        unsigned int t_max, double u_max)
{
    // No running checks here!
    // Set up network variables

    // Set up Neural Network

    // huh, could we change this to decide what kind of distribution we want?
    if(DEBUG)
        printf("creating 2d delay map\n");
    std::vector<std::vector<double>> d_init = initial_delay_vector_2d_map(n_x, n_y, std::abs(n_x / delay_distance), std::abs(n_y / delay_distance));
    // create distance matrix between nodes. This will be spatial in nature
    if(DEBUG)
        printf("creating euclidean point map\n");
    this->point_map = euclidean_point_map(n_x, n_y);
    // Create spatial distance matrix. Not really a lot of information unless combined with delays.
    this->distance_matrix = euclidean_distance_matrix(&(this->point_map), distance_unit);
    // use distance matrix to create initial weight matrix
    // initialize the layer. (n_y / delay_distance) * (n_x / delay_distance)
    this->snn = new BaseSNN(2, d_init, tau_m, u_rest, init_v, t_reset, k_nought, round_zero, alpha, u_max);
    if(DEBUG)
        printf("Passed network initializarion.\n");
    // store hyperparameters
    this->sigma_neighbor = sigma_neighbor;
    this->eta_d = eta_d;
    this->t_max = t_max;
    if(DEBUG)
        printf("Completed initialization.\n");
}

void SNN::train(std::vector<std::vector<double>> X)
{

    // single sample of data. will contain all data samples
    // at some point after training.
    std::vector<double> sample;
    // initialize the sample data to contain data and samples
    sample.resize(2); // this->n_data
    // timeout flag. remains true if no spike happens
    bool time_out;

    // for every training sample...
    for(unsigned int n_sample = 0; n_sample < X.at(0).size(); n_sample++)
    {
        // reset time out flag
        time_out = true;

        // Prepare the sample to be used for training.
        sample.at(0) = X.at(0).at(n_sample);
        sample.at(1) = X.at(1).at(n_sample);


        
        // We will run the same data up to t_max times
        // if no spike is generated, we say that we timed out
        this->snn->re_process(sample);
        // variable used to find minimum distance between
        // input stimulus and the delay vector of jth neuron
        double delay_stimulus_dist = INFINITY;
        // will contain the largest neuron's ID
        unsigned int closest_neuron = UINT32_MAX - 1;
        for(unsigned int t_t = 0; t_t < this->t_max; t_t++)
        {
            // process the current point until we have a spike
            this->snn->process();
            // Is there any spikes?

            // look for any neurons that may have spiked
            if(this->snn->winner_neuron != NO_WINNER)
            {
                unsigned int winner_id = this->snn->winner_neuron;
                /**
                 * We add the time of spike to time table.
                 * We only keep the earliest time of spike
                 * on the time table. It is totally possible for
                 * the spike to spike a second time but not kept
                */
                // We need to compare it to any other neurons that may spike
                // since we need the closest neuron to the input values.
                double x_1 = this->snn->d_ji.at(1).at(winner_id);
                double y_1 = this->snn->d_ji.at(0).at(winner_id); 
                double x_2 = this->snn->stimuli.at(0);
                double y_2 = this->snn->stimuli.at(1);
                double euclidean_dist = sqrt(pow(x_2 - x_1, 2) + pow(y_2 - y_1, 2));

                // is this spiking neuron closer to the input stimulus?
                if(euclidean_dist < delay_stimulus_dist)
                {
                    delay_stimulus_dist = euclidean_dist;
                    closest_neuron = winner_id;
                }
                if(time_out)
                {
                    // we set the time out flag to false. We found at least one spike
                    time_out = false;
                }
            }
        }
        // have we timed out?
        if(time_out)
        {
            // yep. No change in delays.
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

            // there is a winner neuron. Update delays
            for(unsigned int j = 0, d = this->snn->i_layer_size - 1; j < this->snn->i_layer_size; j++, d--)
            {
                // for every input neuron
                if(DEBUG)
                    printf("[ ");
                // you can remove d by chaging order of sample as it is passed.
                for(unsigned int m = 0; m < this->snn->h_layer_size; m++)
                {
                    // for every processing neuron
                    // I know, i do not use 'this'. It is a comple 
                    double delta_mj = eta_d*neighbor(snn->hidden_layer.at(m), snn->hidden_layer.at(closest_neuron))*(sample.at(d) - snn->d_ji.at(j).at(m));
                    this->snn->d_ji.at(j).at(m) += delta_mj;
                    if(DEBUG)
                        printf(" %f ", neighbor(snn->hidden_layer.at(m), snn->hidden_layer.at(closest_neuron)));
                }
                if(DEBUG)
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
    // euclidean distance using the delays as locations.
    double z_mn2 = std::pow(this->snn->d_ji.at(0).at(m.snn_id) - this->snn->d_ji.at(0).at(n.snn_id), 2) + std::pow(this->snn->d_ji.at(1).at(m.snn_id) - this->snn->d_ji.at(1).at(n.snn_id), 2);
    if(DEBUG)
        printf("e^(-%f^2/2*%f^2): %f\n", z_mn2, std::pow(this->sigma_neighbor, 2), std::exp( (-z_mn2) / (2 * std::pow(this->sigma_neighbor, 2)) ));
    return std::exp( (-z_mn2) / (2 * std::pow(this->sigma_neighbor, 2)) );
}




