#include <iostream>
#include "NeuralNetwork.hpp"
#include <vector>
#include <cstdio>

int main()
{
    std::vector<int> topology = {64,64,256,64};

    std::vector<sp::Block> Blocklist = {};
    sp::Block block_0(topology);
    sp::Block block_1(topology);
    sp::Block block_2(topology);
    sp::Block block_3(topology);
    sp::Block block_4(topology);
    sp::Block block_5(topology);
    sp::Block block_6(topology);
    sp::Block block_7(topology);
    sp::Block block_8(topology);
    sp::Block block_9(topology);
    sp::Block block_10(topology);
    sp::Block block_11(topology);

    Blocklist.push_back(block_0);
    Blocklist.push_back(block_1);
    Blocklist.push_back(block_2);
    Blocklist.push_back(block_3);
    Blocklist.push_back(block_4);
    Blocklist.push_back(block_5);
    Blocklist.push_back(block_6);
    Blocklist.push_back(block_7);
    Blocklist.push_back(block_8);
    Blocklist.push_back(block_9);
    Blocklist.push_back(block_10);
    Blocklist.push_back(block_11);


    int num_blks = Blocklist.size();


    sp::SimpleNN nn(Blocklist, 1.0f);


    sp::Matrix2D<float> input (4,300);
    std::fill(input._vals.begin(), input._vals.end(), 0.5);

    std::cout << "training start\n";
    nn.feedForward(input, num_blks);

    // test
    std::vector<float> preds = nn.getPredictions();
    std::cout << "training complete\n";
    std::cout << preds[0] <<','<<preds[1] <<','<<preds[2] <<','<<preds[3] <<','<<preds[4]<< std::endl;


}