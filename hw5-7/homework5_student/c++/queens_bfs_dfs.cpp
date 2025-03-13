#include <iostream>
#include <ctime>

#include "problem/queens.hpp"

#include "algorithm/depth_first_search.hpp"
#include "algorithm/breadth_first_search.hpp"

int main(){
    std::ios::sync_with_stdio(false);

    // time_t t0 = time(nullptr);
    
    clock_t start = clock();
    
    QueensState state(8);

    // BreadthFirstSearch<QueensState> bfs(state);
    // bfs.search(true, false);

    DepthFirstSearch<QueensState> dfs(state);
    dfs.search(true, false);
    
    // std::cout << time(nullptr) - t0 << std::endl;

    std::cout << "time = " << (double)(clock() - start)/CLOCKS_PER_SEC << "s" << std::endl;

    std::cout << CLOCKS_PER_SEC << std::endl;

    return 0;
}
