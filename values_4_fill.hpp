//
// Created by 12192 on 2021/10/27.
//
#include <iostream>
#include <vector>
#include "rapidcsv.h"

//pos_embedding, (301, 64)
std::vector<float> posit {};
//cls_token, (1, 64)
std::vector<float> cls {};
//init weight, (64, 4)
std::vector<float> ini_w {};
//init bias, (64)
std::vector<float> ini_b {};

//BLOCK
//block_0
std::vector<float> blk0_norm_w0 {}; //(64)
std::vector<float> blk0_norm_b0 {}; //(64)
std::vector<float> blk0_norm_w1 {}; //(64)
std::vector<float> blk0_norm_b1 {}; //(64)

std::vector<float> blk0_w0 {}; // to_qkv, No bias (64,64)
std::vector<float> blk0_w1 {}; //(64,256)
std::vector<float> blk0_b1 {}; //(64,1)
std::vector<float> blk0_w2 {}; //(256,64)
std::vector<float> blk0_b2 {}; //(256,1)

//block_1
std::vector<float> blk1_norm_w0 {};
std::vector<float> blk1_norm_b0 {};
std::vector<float> blk1_norm_w1 {};
std::vector<float> blk1_norm_b1 {};

std::vector<float> blk1_w0 {}; // to_qkv, No bias
std::vector<float> blk1_w1 {};
std::vector<float> blk1_b1 {};
std::vector<float> blk1_w2 {};
std::vector<float> blk1_b2 {};

//block_2
std::vector<float> blk2_norm_w0 {};
std::vector<float> blk2_norm_b0 {};
std::vector<float> blk2_norm_w1 {};
std::vector<float> blk2_norm_b1 {};

std::vector<float> blk2_w0 {}; // to_qkv, No bias
std::vector<float> blk2_w1 {};
std::vector<float> blk2_b1 {};
std::vector<float> blk2_w2 {};
std::vector<float> blk2_b2 {};

//block_3
std::vector<float> blk3_norm_w0 {};
std::vector<float> blk3_norm_b0 {};
std::vector<float> blk3_norm_w1 {};
std::vector<float> blk3_norm_b1 {};

std::vector<float> blk3_w0 {}; // to_qkv, No bias
std::vector<float> blk3_w1 {};
std::vector<float> blk3_b1 {};
std::vector<float> blk3_w2 {};
std::vector<float> blk3_b2 {};

//block_4
std::vector<float> blk4_norm_w0 {};
std::vector<float> blk4_norm_b0 {};
std::vector<float> blk4_norm_w1 {};
std::vector<float> blk4_norm_b1 {};

std::vector<float> blk4_w0 {}; // to_qkv, No bias
std::vector<float> blk4_w1 {};
std::vector<float> blk4_b1 {};
std::vector<float> blk4_w2 {};
std::vector<float> blk4_b2 {};

//block_5
std::vector<float> blk5_norm_w0 {};
std::vector<float> blk5_norm_b0 {};
std::vector<float> blk5_norm_w1 {};
std::vector<float> blk5_norm_b1 {};

std::vector<float> blk5_w0 {}; // to_qkv, No bias
std::vector<float> blk5_w1 {};
std::vector<float> blk5_b1 {};
std::vector<float> blk5_w2 {};
std::vector<float> blk5_b2 {};

//block_6
std::vector<float> blk6_norm_w0 {};
std::vector<float> blk6_norm_b0 {};
std::vector<float> blk6_norm_w1 {};
std::vector<float> blk6_norm_b1 {};

std::vector<float> blk6_w0 {}; // to_qkv, No bias
std::vector<float> blk6_w1 {};
std::vector<float> blk6_b1 {};
std::vector<float> blk6_w2 {};
std::vector<float> blk6_b2 {};

//block_7
std::vector<float> blk7_norm_w0 {};
std::vector<float> blk7_norm_b0 {};
std::vector<float> blk7_norm_w1 {};
std::vector<float> blk7_norm_b1 {};

std::vector<float> blk7_w0 {}; // to_qkv, No bias
std::vector<float> blk7_w1 {};
std::vector<float> blk7_b1 {};
std::vector<float> blk7_w2 {};
std::vector<float> blk7_b2 {};

//block_8
std::vector<float> blk8_norm_w0 {};
std::vector<float> blk8_norm_b0 {};
std::vector<float> blk8_norm_w1 {};
std::vector<float> blk8_norm_b1 {};

std::vector<float> blk8_w0 {}; // to_qkv, No bias
std::vector<float> blk8_w1 {};
std::vector<float> blk8_b1 {};
std::vector<float> blk8_w2 {};
std::vector<float> blk8_b2 {};

//block_9
std::vector<float> blk9_norm_w0 {};
std::vector<float> blk9_norm_b0 {};
std::vector<float> blk9_norm_w1 {};
std::vector<float> blk9_norm_b1 {};

std::vector<float> blk9_w0 {}; // to_qkv, No bias
std::vector<float> blk9_w1 {};
std::vector<float> blk9_b1 {};
std::vector<float> blk9_w2 {};
std::vector<float> blk9_b2 {};

//block_10
std::vector<float> blk10_norm_w0 {};
std::vector<float> blk10_norm_b0 {};
std::vector<float> blk10_norm_w1 {};
std::vector<float> blk10_norm_b1 {};

std::vector<float> blk10_w0 {}; // to_qkv, No bias
std::vector<float> blk10_w1 {};
std::vector<float> blk10_b1 {};
std::vector<float> blk10_w2 {};
std::vector<float> blk10_b2 {};

//block_11
std::vector<float> blk11_norm_w0 {};
std::vector<float> blk11_norm_b0 {};
std::vector<float> blk11_norm_w1 {};
std::vector<float> blk11_norm_b1 {};

std::vector<float> blk11_w0 {}; // to_qkv, No bias
std::vector<float> blk11_w1 {};
std::vector<float> blk11_b1 {};
std::vector<float> blk11_w2 {};
std::vector<float> blk11_b2 {};
