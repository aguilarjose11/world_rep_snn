#include <iostream>
#include <armadillo>
#include <vector>
#include "ou_snn.h"

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
    return 0;
}