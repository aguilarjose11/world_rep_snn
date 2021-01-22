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


std::vector<int> WorldRep::dijkstra(
    std::vector<std::vector<double>> src, 
std::vector<std::vector<double>> end)
{
    std::vector<std::vector<double>> graph = world_rep;

    unsigned int V = graph.size();
    unsigned int n_points = graph.at(0).size();

    arma::Mat<double> dist_map_arma = euclidean_distance_matrix(&graph, 1);
    // to contain the distance map.
    std::vector<std::vector<double>> dist_map(dist_map_arma.n_rows, std::vector<double>(dist_map_arma.n_cols, 0));
    // convert arma::Mat<double> to std::vector matrix
    for (size_t i = 0; i < dist_map_arma.n_rows; ++i) 
    {
        for(size_t p = 0; p < dist_map_arma.n_cols; p++)
        {
            dist_map.at(i).at(p) = dist_map_arma(i, p);
        }
    };

    unsigned int start, goal;

    double min_dist_start = INFINITY, min_dist_goal = INFINITY;

    // Calculate closest points on map for given.
    start = (UINT32_MAX) - 1;
    goal = (UINT32_MAX) - 1;
    for(int i = 0; i < n_points; i++)
    {
        double p2p_src = std::sqrt(std::pow((graph.at(0).at(i) - src.at(0).at(0)), 2) + std::pow((graph.at(1).at(i) - src.at(1).at(0)), 2));
        if(p2p_src < min_dist_start)
        {
            min_dist_start = std::sqrt(std::pow((graph.at(0).at(i) - src.at(0).at(0)), 2) + std::pow((graph.at(1).at(i) - src.at(1).at(0)), 2));
            start = i;
        }
        double p2p_goal = std::sqrt(std::pow((graph.at(0).at(i) - end.at(0).at(0)), 2) + std::pow((graph.at(1).at(i) - end.at(1).at(0)), 2));
        if(p2p_goal < min_dist_goal)
        {
            min_dist_goal = std::sqrt(std::pow((graph.at(0).at(i) - end.at(0).at(0)), 2) + std::pow((graph.at(1).at(i) - end.at(1).at(0)), 2));
            goal = i;
        }
    }

    

    // The output array. dist[i] 
    // will hold the shortest 
    // distance from src to i 
    std::vector<int> dist;
    dist.resize(V);
  
    // sptSet[i] will true if vertex 
    // i is included / in shortest 
    // path tree or shortest distance  
    // from src to i is finalized 
    bool sptSet[V]; 
  
    // Parent array to store 
    // shortest path tree 
    std::vector<int> parent;
    parent.resize(V);
  
    // Initialize all distances as  
    // INFINITE and stpSet[] as false 
    for (unsigned int i = 0; i < V; i++) 
    { 
        parent.at(0) = -1; 
        dist.at(i) = INT_MAX; 
        sptSet[i] = false; 
    } 
  
    // Distance of source vertex  
    // from itself is always 0 
    dist.at(start) = 0; 
  
    // Find shortest path 
    // for all vertices 
    for (unsigned int count = 0; count < V - 1; count++) 
    { 
        // Pick the minimum distance 
        // vertex from the set of 
        // vertices not yet processed.  
        // u is always equal to src 
        // in first iteration. 
        int u = minDistance(dist, sptSet, V); 
  
        // Mark the picked vertex  
        // as processed 
        sptSet[u] = true; 
  
        // Update dist value of the  
        // adjacent vertices of the 
        // picked vertex. 
        for (int v = 0; v < V; v++) 
  
            // Update dist[v] only if is 
            // not in sptSet, there is 
            // an edge from u to v, and  
            // total weight of path from 
            // src to v through u is smaller 
            // than current value of 
            // dist[v] 
            if (!sptSet[v] && graph.at(u).at(v) && 
                dist.at(u) + graph.at(u).at(v) < dist.at(v)) 
            { 
                parent.at(v) = u; 
                dist.at(v) = dist.at(u) + graph.at(u).at(v); 
            }  
    } 
  
    // print the constructed 
    // distance array 
    // Create vector containing path:
    std::vector<int> CURRENT_PATH;
    CURRENT_PATH.clear();
    printPath(parent, &CURRENT_PATH, goal);
    return CURRENT_PATH;

}


void WorldRep::printPath(std::vector<int> parent, std::vector<int> *final_path, int j) 
{ 
      
    // Base Case : If j is source 
    if (parent[j] == - 1) 
        return; 
  
    printPath(parent, final_path, parent.at(j)); 
  
    final_path->push_back(j); 
} 

int WorldRep::minDistance(std::vector<int> dist, bool sptSet[], unsigned int V) 
{ 
      
    // Initialize min value 
    int min = INT_MAX, min_index; 
  
    for (int v = 0; v < V; v++) 
        if (sptSet[v] == false && 
                   dist[v] <= min) 
            min = dist[v], min_index = v; 
  
    return min_index; 
}

double WorldRep::angle_2d(std::vector<double> n_1, std::vector<double> n_2)
{
    double dx = n_2.at(0) - n_1.at(0);
    double dy = n_2.at(1) - n_1.at(1);

    double value = std::atan2(dx, dy);

    if(value > 0)
    {
        return value * (180/M_PI);
    }
    else
    {
        return (2*M_PI + value) * 360 / (2*M_PI);
    }
}



std::vector<std::vector<double>> WorldRep::distance_map(std::vector<double> X, std::vector<double> Y)
{
    double angle = 90, max_dist = 2;
    unsigned int angle_n = (unsigned int)(360 / angle + 1);
    std::vector<unsigned int> angle_list;
    for(int a = 0; a < angle_n; a++)
    {
        angle_list.push_back(angle*a);
    }
    if(angle_list.back() < 360)
    {
        angle_list.push_back(360);
    }

    int node;
    std::vector<std::vector<double>> distance_map(X.size(), std::vector<double>(Y.size(), 0.0));
    std::vector<double> node_vertex, vertex;
    std::deque<int> visited, processing;

    processing.push_back(0);

    while(!processing.empty())
    {
        node = processing.front();
        processing.pop_front();
        visited.push_back(node);
        node_vertex = std::vector<double>({X.at(node), Y.at(node)});
        for(unsigned int a = 0, b = 1; b < angle_list.size(); a++, b++)
        {
            double lower_bound = angle_list.at(a);
            double upper_bound = angle_list.at(b);
            double min_dist = INFINITY;
            unsigned int closest_node = UINT32_MAX - 1;

            for(unsigned int i = 0; i < X.size(); i++)
            {
                if(i == node)
                {
                    continue;
                }

                std::vector<double> curr_vertex({X.at(i), Y.at(i)});
                double vertex_angle = angle_2d(curr_vertex, node_vertex);
                if(lower_bound <= vertex_angle || vertex_angle < upper_bound)
                {
                    double x_1 = curr_vertex.at(0), y_1 = curr_vertex.at(1);
                    double x_2 = node_vertex.at(0), y_2 = node_vertex.at(1);
                    double p2p_distance = sqrt(pow(x_2 - x_1, 2) + pow(y_2 - y_1, 2));
                    if(p2p_distance < min_dist)
                    {
                        min_dist = p2p_distance;
                        closest_node = i;
                    }
                }
            }
            if(std::find(visited.begin(), visited.end(), closest_node) != visited.end() || min_dist > max_dist)
            {
                closest_node = UINT32_MAX - 1;
            }
            if(closest_node != UINT32_MAX - 1)
            {
                distance_map.at(closest_node).at(node) = (double) min_dist;
                distance_map.at(node).at(closest_node) = (double) min_dist;
                processing.push_back(closest_node);
            }
        }
    }
    return distance_map;

}