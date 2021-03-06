#include "libsnnlfisrm/snn.h"
#include <queue>

#define MAP_DIMENSIONS 2
#define CONST_POTENTIAL 0
#define REFRACTORY_TIME 0
#define ALPHA_ENCODING 1
#define X_ (unsigned int)1
#define Y_  (unsigned int)0

/**
 * World Representation and path calculator.
 * 
 * Using a Spiking Neural Network as described in libsnnlfisrm 
 * library, a given path (for offline training only) is used
 * to create a world representation that can be used alongside
 * a path planning algorithm to find the most optimal path from
 * a certain point to another.
 * 
 * @author Jose E. Aguilar Escamilla
 * January 19, 2021
*/
class WorldRep
{
    public:
        
        /**
         * Assumptions:
         * - Map dimensions will always be 2.
         * - u_rest will remain 0.
         * - no initial threshold, only fixed.
         * - t_reset always 0.
         * - u_max mostly used for aesthetics. no real impact.
         * - alpha is assumed to always be 1
         * - Origin is always 0
         * 
         * @param k_nought Initial height of spike.
         * @param tau_m Membrane decay constant.
         * @param threshold Spiking threshold.
         * @param zero_cutoff Cutoff at which membrane activity
         *                    is considered for nought.
         * @param n_x Maximum x value for map.
         * @param n_y Maximum y value for map.
         * @param delay_distance Distance between all neurons.
         * @param sigma_neighbor Neighborhood's function constant.
         * @param eta_d Learning rate/strength.
         * @param t_max Total running time per simulation cycle.
         * @param u_max Potential reached during spiking.
         * @param prune_dist Distance cutoff for pruning neurons.
        */
        WorldRep(unsigned int k_nought, double tau_m, 
        double threshold, double zero_cutoff, unsigned int n_x, 
        unsigned int n_y, double delay_distance, 
        double sigma_neighbor, double eta_d, unsigned int t_max, 
        double u_max, double prune_dist);

        typedef enum PathAlgorithm
        {
            dijkstras,
            a_stars // Not implemented yet
        } PathAlgorithm;

        /**
         * Train aformentioned SNN and prune.
        */
        void train(std::vector<std::vector<double>> data);
        
        /**
         * Utilize an algorithm for finding best path.
        */
        std::vector<std::vector<double>> get_path(PathAlgorithm algo,
        std::vector<double> src, std::vector<double> goal);
        
        /**
         * Accesor function for the map representation.
        */
        std::vector<std::vector<double>> get_map();

        std::vector<unsigned int> dijkstra(unsigned int src, unsigned int j);
        SNN snn;

    private:
    
        /**
         * Euclidean distance following 2d version for snn 
         * distance vectors to compare b.
        */
       double euclidean_difference(unsigned int m);

        /**
         * Remove the neurons that have moved less than the
         * indicated minimum value.
        */
        std::vector<std::vector<double>> prune();

        void printPath(std::vector<int> parent, 
        std::vector<unsigned int> *final_path, int j);

        int minDistance(std::vector<int> dist, std::vector<bool>sptSet, 
        unsigned int V);

        double angle_2d(std::vector<double> n_1, std::vector<double> n_2);

        std::vector<std::vector<double>> distance_map(std::vector<double> X, std::vector<double> Y);


        // variables
        std::vector<std::vector<double>> world_rep;

        std::vector<std::vector<double>> distance_matrix;
        double prune_dist;
};