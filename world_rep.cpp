#include "world_rep.h"

WorldRep::WorldRep(unsigned int k_nought, double tau_m, 
double threshold, double zero_cutoff, unsigned int n_x, 
unsigned int n_y, double delay_distance, double sigma_neighbor, 
double eta_d, unsigned int t_max, double u_max, double prune_dist)
{
    world_rep.resize(MAP_DIMENSIONS);
    this->prune_dist = prune_dist;
    if(DEBUG)
    {
        printf("Variable Initialization succesful.\n");
    }
    this->snn = SNN(MAP_DIMENSIONS, tau_m, CONST_POTENTIAL, threshold,
                    REFRACTORY_TIME, k_nought, zero_cutoff, 
                    ALPHA_ENCODING, n_x, n_y, delay_distance, 0,
                    sigma_neighbor, eta_d, t_max, u_max);
    if(DEBUG)
    {
        printf("SNN model Initialization succesful.\n");
    }
}


void WorldRep::train(std::vector<std::vector<double>> data)
{
    snn.train(data);
    world_rep = prune();
}


double WorldRep::euclidean_difference(unsigned int m)
{
    double x_1, y_1, x_2, y_2, euclidean_distance;

    x_1 = snn.snn->d_ji.at(X_).at(m);
    y_1 = snn.snn->d_ji.at(Y_).at(m); 
    x_2 = snn.snn->d_ji_reset.at(X_).at(m);
    y_2 = snn.snn->d_ji_reset.at(Y_).at(m);
    euclidean_distance = sqrt(pow(x_2 - x_1, 2) + pow(y_2 - y_1, 2));
    
    return euclidean_distance;
}


std::vector<std::vector<double>> WorldRep::prune()
{
    std::vector<std::vector<double>> neural_map;
    neural_map.resize(2);

    // only keep neurons that moved a lot.
    for(unsigned int neuron = 0; neuron < snn.snn->d_ji.at(0).size(); neuron++)
    {
        if(euclidean_difference(neuron) > prune_dist)
        {
            neural_map.at(0).push_back(snn.snn->d_ji.at(X_).at(neuron));
            neural_map.at(1).push_back(snn.snn->d_ji.at(Y_).at(neuron));
        }
    }

    return neural_map;
}


std::vector<std::vector<double>> WorldRep::get_map()
{
    return world_rep;
}

std::vector<std::vector<double>> WorldRep::get_path()
{
    
}
