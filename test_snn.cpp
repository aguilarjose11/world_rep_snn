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
    double init_v = 20; // when testing math, make this large
    unsigned char t_rest = 2;
    double kappa_naugh = 3;
    double round_zero = 0.1;
    double u_max = 10;

    SpikeResponseModelNeuron single_neuron = SpikeResponseModelNeuron(
        snn_id, n_inputs, init_d, tau_m, u_rest,
        init_v, t_rest, kappa_naugh, round_zero, u_max);
    std::cout << "Created individual neuron." << std::endl;

    std::vector<std::vector<bool>> presynaptic_train = {
        {0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
    };


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
    exp_res << "]" << std::endl;

    printf("Testing FSTNs\n");
    double alpha = 1;
    FirstSpikeTimeNeuron fstn = FirstSpikeTimeNeuron(0, alpha);
    fstn.dendrite = 2;
    fstn.encode();

    unsigned char total_pulses = 10;

    exp_res << "First Spike Time Neuron spike after " << total_pulses << "pulses." << std::endl << "[";
    exp_res << fstn.axon.signal << ", ";
    for(unsigned char pulse= 0; pulse < total_pulses; pulse++)
    {
        fstn.t_pulse();
        exp_res << fstn.axon.signal << ", ";
    }
    exp_res << "]" << std::endl;

    exp_res.close();

    // Testing training algorthm
    unsigned int n_data = 2;
    tau_m = 0.4;
    u_rest = 0;
    init_v = 5.5;
    unsigned int t_reset = 3;
    double k_nought = 3;
    round_zero = 0.05;
    alpha = 1;
    // note that n_x * n_y = h_layer_size
    unsigned int n_x = 8;
    unsigned int n_y = 8;
    double delay_distance = 0.5;
    unsigned int distance_unit = 1;
    double sigma_neighbor = 1;
    double eta_d = .5;
    unsigned int t_max = 25;
    u_max = 10;

    printf("Create SNN\n");
    SNN model(n_data, tau_m, u_rest, init_v, 
    t_reset, k_nought, round_zero, alpha, n_x, n_y, delay_distance,
    distance_unit, sigma_neighbor, eta_d, t_max, u_max);

    std::vector<std::vector<double>> data = {
        {0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0}
    };
    std::vector<std::vector<double>> data_2 = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1},
    };

    printf("Create output file\n");

    // save delays into file:
    std::ofstream delay_file;
    delay_file.open("training_delays.txt");
    
    // print out delays
    delay_file << "Delays before training:" << std::endl;
    for(unsigned int i = 0; i < model.snn->d_ji.size(); i++)
    {
        delay_file << "[ ";
        for(unsigned int j = 0; j < model.snn->d_ji.at(i).size(); j++)
        {
            delay_file << model.snn->d_ji.at(i).at(j) << ", ";
        }
        delay_file << "]" << std::endl;
    }
    // print out weights
   
    for(unsigned int p = 0; p < 1; p++)
        model.train(data);

    delay_file << std::endl << "Delays after training:"<< std::endl;
     for(unsigned int i = 0; i < model.snn->d_ji.size(); i++)
    {
        delay_file << "[ ";
        for(unsigned int j = 0; j < model.snn->d_ji.at(i).size(); j++)
        {
            delay_file << model.snn->d_ji.at(i).at(j) << ", ";
        }
        delay_file << "]" << std::endl;
    }


    for(unsigned int p = 0; p < 8; p++, sigma_neighbor -= 0.1)
    {
        model = SNN(n_data, tau_m, u_rest, init_v, 
        t_reset, k_nought, round_zero, alpha, n_x, n_y, delay_distance,
        distance_unit, sigma_neighbor, eta_d, t_max, u_max);
        
        model.train(data);

        delay_file << std::endl << "Delays after training with sigma_neighbor = " << sigma_neighbor << ": \n" << std::endl;
        for(unsigned int i = 0; i < model.snn->d_ji.size(); i++)
        {
            delay_file << "[ ";
            for(unsigned int j = 0; j < model.snn->d_ji.at(i).size(); j++)
            {
                delay_file << model.snn->d_ji.at(i).at(j) << ", ";
            }
            delay_file << "]" << std::endl;
        }
    }

    delay_file.close();
    

}