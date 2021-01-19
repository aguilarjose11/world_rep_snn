#include <iostream>
#include <armadillo>
#include <vector>
#include <cstdlib>
#include <fstream>
#include "libsnnlfisrm/snn.h"

/**
 * Calculates initial delays
*/
arma::Col<double> get_initial_delays();

/**
 * Calculates initial weights for lateral synapses
*/
arma::Col<double> get_initial_weights();

int main(int argc, const char **argv) {
    
    // Test individual LIF/SRM neuron
    int snn_id = 2;
    int n_inputs = 2;
    int n_lateral = 1;
    std::vector<double> init_d({2, 4});
    std::vector<double> init_w({4});
    double tau_m = 1;
    double u_rest = 3;
    double init_v = 20;
    unsigned char t_rest = 2;
    double kappa_naugh = 3;
    double round_zero = 0.1;
    double u_max = 10;

    SpikeResponseModelNeuron neuron(snn_id, n_inputs, init_d, tau_m, 
    u_rest, init_v, t_rest, kappa_naugh, round_zero, u_max);

    printf("delay vector: [");
    for(unsigned int i = 0; i < neuron.d_j.size(); i++)
    {
        printf(" %f ", neuron.d_j.at(i));
    }
    printf("]\n");

    // test t_pulse is working and making kappas...
    std::vector<double> u;
    printf("number of kappas: %d\n", (int)neuron.k_filter_list.at(K_LIST_INPUT_SYNAPSES).size());
    neuron.dendrite = std::vector<DelayedSpike>({DelayedSpike(0, true), DelayedSpike(0, 0)});
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.dendrite = std::vector<DelayedSpike>({DelayedSpike(0, false), DelayedSpike(0, 0)});
    neuron.t_pulse();
    u.push_back(neuron.membrane_potential());
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.dendrite = std::vector<DelayedSpike>({DelayedSpike(0, true), DelayedSpike(0, 0)});
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("number of kappas: %d\n", (int)neuron.k_filter_list.at(K_LIST_INPUT_SYNAPSES).size());

    printf("u plot: [ %f", u.at(0));
    for(unsigned int v = 1; v < u.size(); v++)
        printf(", %f", u.at(v));
    printf("]\n");

    // Testing FSTN
    printf("Testing Fisrt spike time neuron\n");
    double alpha = 1;
    FirstSpikeTimeNeuron fstn(0, alpha);
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)2;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();
    fstn.dendrite = (double)5;
    fstn.t_pulse();


    // Testing neural network
    // parameters:
    unsigned int i_layer_size = 2;
    unsigned int h_layer_size = 16;
    std::vector<std::vector<double>> d_init({
        {4, 3, 2, 3, 4, 3, 3, 2, 1, 4, 3, 3, 3, 2, 5, 2}, 
        {1, 4, 2, 4, 6, 2, 1, 3, 5, 3, 3, 3, 3, 2, 5, 2}
    });
    std::vector<std::vector<double>> w_init({
        {1, 3, 2, -1, 3, 1, -2, 1, -2, 1, 3, 3, 3, 2, 5, 2},
        {3, 4, 2, 3, 4, 12, 1, 1, 2, 1, 3, 3, 3, 2, 5, 2},
        {2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 3, 3, 3, 2, 5, 2},
        {-1, -2, -1, -2, -3, -4, -9, -1, 2, -2, 3, 3, 3, 2, 5, 2},
        {3, 4, 2, 1, 3, 6, 3, 1, 2, 3, 3, 3, 3, 2, 5, 2},
        {1, 3, 1, 3, 5, 3, 1, 0, 3, 1, 3, 3, 3, 2, 5, 2},
        {-2, -3, -1, -3, -5, 3, 2, 4, 5, 4, 3, 3, 3, 2, 5, 2},
        {1, 6, 2, 2, 4, 6, 3, 2, 4, 5, 3, 3, 3, 2, 5, 2},
        {-2, -5, -1, 4, -1, -3, 2, -7, -1, 3, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
        {1, 5, 7, 2, 4, 6, 4, 1, 3, 5, 3, 3, 3, 2, 5, 2},
    });
    tau_m = 4;
    u_rest = 0;
    init_v = 20;
    double t_reset = 3;
    double k_nought = 10;
    round_zero = 0.1; 
    alpha = 2;

    BaseSNN snn(i_layer_size, d_init, tau_m, u_rest, init_v, t_reset, k_nought, 
    round_zero, alpha, u_max);

    snn.re_process({0, 0});
    for(int p = 0; p < 20; p++)
    {
        snn.process();
        for(unsigned int i = 0; i < h_layer_size; i++)
        {
            double u_i = snn.hidden_layer.at(i).membrane_potential();
            printf("U_%u = %f\n", i, u_i);
            if(snn.hidden_layer.at(i).axon.signal)
            {
                // this neuron is the "winner"
                printf("Spike %u winner (Reached threshold)\n", i);
                goto next_test;
            }
        }
    }
    next_test:
    // Test euclidean_distance_matrix function
    double distance_unit = 2;

    printf("Testing matrices.\n");
    arma::Mat<double> e_mat = euclidean_distance_matrix (&snn, distance_unit);
    printf("Matrix Content\n\n");
    for(unsigned int i = 0; i < snn.h_layer_size; i++)
    {
        for(unsigned int j = 0; j < snn.h_layer_size; j++)
        {
            printf("%f\t", e_mat(i, j));
        }
        printf("\n");
    }


    // Test euclidean initial weight function
    double sigma_1 = .5;
    double sigma_2 = 2;
    std::vector<arma::Col<double>> w_mat = initial_weight_euclidean(e_mat, sigma_1, sigma_2);
    printf("\nPrinting weight matrix's contents:\n\n");
    printf("Weight Matrix Content\n\n");
    for(unsigned int i = 0; i < snn.h_layer_size; i++)
    {
        for(unsigned int j = 0; j < snn.h_layer_size; j++)
        {
            printf("%f\t", w_mat.at(i).at(j));
        }
        printf("\n");
    }

 
    // Testing euclidean_distance_matrix without snn
    std::vector<std::vector<double>> point_list({
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
        {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}
    });

    

    printf("Testing matrices.\n");
    arma::Mat<double> e_mat_init = euclidean_distance_matrix(&point_list, distance_unit);
    printf("Matrix Content\n\n");
    for(unsigned int i = 0; i < e_mat_init.n_rows; i++)
    {
        for(unsigned int j = 0; j < e_mat_init.n_cols; j++)
        {
            printf("%f\t", e_mat_init(i, j));
        }
        printf("\n");
    }

    // Testing Random delay generator
    printf("\nTesting Random Delay generator\n");
    std::vector<std::vector<double>> initial_delays;

    unsigned int n_x = 10;
    unsigned int n_y = 10;
    unsigned int delays_per_row = 10;
    unsigned int delays_per_column = 10;

    initial_delays = initial_delay_vector_2d_map(n_x, n_y, 
    delays_per_row, delays_per_column);

    printf("Delay Matrix Content\n\n");
    for(unsigned int i = 0; i < initial_delays.size(); i++)
    {
        for(unsigned int j = 0; j < initial_delays.at(i).size(); j++)
        {
            printf("%f\t", initial_delays.at(i).at(j));
        }
        printf("\n");
    }

    // Test point map creator
    printf("\nTesting point map creator.\n");
    unsigned int x_size = 4;
    unsigned int y_size = 7;
    std::vector<std::vector<double>> map_ = euclidean_point_map(x_size, y_size);
    printf("Point Map Matrix Content\n\n");

    for(unsigned int j = 0; j < x_size*y_size; j++)
    {
        printf("(%f, %f)\t", map_.at(0).at(j), map_.at(1).at(j));
    }


    // Testing training algorthm
    unsigned int n_data = 2;
    tau_m = 1.5;
    u_rest = 2;
    init_v = 4.5;
    t_reset = 3;
    k_nought = 2.5;
    round_zero = 0.05;
    alpha = 4;
    // note that n_x * n_y = h_layer_size
    n_x = 10;
    n_y = 10;
    double delay_distance = 1;
    distance_unit = 1;
    sigma_1 = 0.7;
    sigma_2 = 1.6;

    double sigma_neighbor = 1;

    double eta_d = .5;
    unsigned int t_max = 25;
    unsigned int t_delta = 3;
    double ltd_max = -0.45;
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

    // save delays into file:
    std::ofstream delay_file;
    delay_file.open("snn_data_delays.txt");
    
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
    delay_file.close();
    


    // Armadillo version printout
    arma::arma_version ver;
    printf("\nArmadillo Version: %s\n", ver.as_string().c_str());
    return 0;
}


