#include <iostream>
#include <armadillo>
#include <vector>
#include "ou_snn.h"

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
    arma::Col<double> init_d({2, 4});
    arma::Col<double> init_w({4});
    double tau_m = 1;
    double u_rest = 3;
    double init_v = 20;
    unsigned char t_rest = 2;
    double kappa_naugh = 3;
    double round_zero = 0.1;

    OU_LIF_SRM neuron(snn_id, n_inputs, n_lateral, init_d, init_w, tau_m, 
    u_rest, init_v, t_rest, kappa_naugh, round_zero);

    printf("delay vector: [");
    for(unsigned int i = 0; i < neuron.d_j.size(); i++)
    {
        printf(" %f ", neuron.d_j.at(i));
    }
    printf("]\n");

    // test t_pulse is working and making kappas...
    std::vector<double> u;
    printf("number of kappas: %d\n", (int)neuron.k_filter_list.at(K_LIST_INPUT_SYNAPSES).size());
    neuron.dendrite = std::vector<D_Spike>({D_Spike(0, true), D_Spike(0, 0)});
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
    neuron.dendrite = std::vector<D_Spike>({D_Spike(0, false), D_Spike(0, 0)});
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
    neuron.dendrite = std::vector<D_Spike>({D_Spike(0, true), D_Spike(0, 0)});
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    printf("membrane potential: %f\n", neuron.membrane_potential());
    u.push_back(neuron.membrane_potential());
    neuron.t_pulse();
    neuron.horizontal_dendrite = std::vector<D_Spike>({D_Spike(0, true), D_Spike(0, true)});
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
    printf("Testing Fisrt spike time neuron");
    double alpha = 1;
    OU_FSTN fstn(0, alpha);
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
    unsigned int h_layer_size = 10;
    std::vector<arma::Col<double>> d_init({
        {4, 3, 2, 3, 4, 3, 3, 2, 1, 4}, 
        {1, 4, 2, 4, 6, 2, 1, 3, 5, 3}
    });
    std::vector<arma::Col<double>> w_init({
        {1, 3, 2, -1, 3, 1, -2, 1, -2, 1},
        {2, 4, 2, 3, 4, 12, 1, 1, 2, 1},
        {2, 3, 2, 1, 2, 1, 2, 1, 1, 2},
        {-1, -2, -1, -2, -3, -4, -9, -1, 2, -2},
        {2, 4, 2, 1, 3, 6, 3, 1, 2, 3},
        {1, 3, 1, 3, 5, 3, 1, 0, 3, 1},
        {-3, -3, -1, -3, -5, 3, 2, 4, 5, 4},
        {4, 6, 2, 2, 4, 6, 3, 2, 4, 5},
        {-3, -5, -1, 4, -1, -3, 2, -7, -1, 3},
        {2, 5, 7, 2, 4, 6, 4, 1, 3, 5},
    });
    tau_m = 4;
    u_rest = 0;
    init_v = 20;
    double t_reset = 3;
    double k_nought = 10;
    round_zero = 0.1; 
    alpha = 2;

    OU_SRM_NET snn(i_layer_size, h_layer_size, d_init, w_init,
    tau_m, u_rest, init_v, t_reset, k_nought, round_zero, alpha);

    for(int p = 0; p < 20; p++)
    {
        snn.process({0, 0});
        for(unsigned int i = 0; i < h_layer_size; i++)
        {
            double u_i = snn.hidden_layer.at(i).membrane_potential();
            printf("U_%u = %f\n", i, u_i);
            if(snn.hidden_layer.at(i).axon.signal)
            {
                // this neuron is the "winner"
                printf("Spike %u winner (Reached threshold)\n", i);
                return 0;
            }
        }
    }



    return 0;
}