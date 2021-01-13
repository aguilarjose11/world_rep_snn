#include <iostream>
#include <armadillo>
#include <vector>
#include <cstdlib>
#include <fstream>

#include "libsnnlfisrm/snn.h"

int main(int argc, const char **argv)
{
    // Values for single neuron.
    int snn_id = 2;
    int n_inputs = 2;
    int n_lateral = 1;
    arma::Col<double> init_d({2, 4});
    arma::Col<double> init_w({4});
    double tau_m = 2;
    double u_rest = 0;
    double init_v = 4; // when testing math, make this large
    unsigned char t_rest = 1;
    double kappa_naugh = 3;
    double round_zero = 0.1;
    double u_max = 10;

    SpikeResponseModelNeuron single_neuron = SpikeResponseModelNeuron(
        snn_id, n_inputs, n_lateral, init_d, init_w, tau_m, u_rest,
        init_v, t_rest, kappa_naugh, round_zero, 0, 0, u_max
    );
    std::cout << "Created individual neuron." << std::endl;

    std::vector<std::vector<bool>> presynaptic_train = {
        {0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
    };

    single_neuron.horizontal_dendrite.at(0) = DelayedSpike();

    std::ofstream exp_res;
    exp_res.open("experiment_results.log");
    exp_res << "[";
    for(unsigned int epoch = 0; epoch < presynaptic_train.at(0).size(); epoch++)
    {
        exp_res << single_neuron.membrane_potential() << ", ";
        // load dendrites with current value
        single_neuron.dendrite.at(0) = DelayedSpike(0, presynaptic_train.at(0).at(epoch));
        single_neuron.dendrite.at(1) = DelayedSpike(0, presynaptic_train.at(1).at(epoch));

        // Move network forward
        single_neuron.t_pulse();
        
    }
    exp_res << "]";
    exp_res.close();
    

}