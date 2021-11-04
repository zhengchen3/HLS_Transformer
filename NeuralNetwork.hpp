//
// Created by 12192 on 2021/9/25.
//
#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

//#include "params.h"
//#include "rapidcsv.h"


namespace sp {

    //simple activation function
    inline float Sigmoid(float x) {
        return 1.0f / (1 + exp(-x));
    }

    //derivative of activation function
    // x = sigmoid(input);
    inline float DSigmoid(float x) {
        return (x * (1 - x));
    }

    // multi-label activation
    inline Matrix2D<float> Softmax(sp::Matrix2D<float> input){
        sp::Matrix2D<float> exp_input(input._cols, input._rows);
        for(int i=0; i<input._vals.size(); i++){
            exp_input._vals[i] = exp(input._vals[i]);
        }
        float sum_of_exp = std::accumulate(exp_input._vals.begin(), exp_input._vals.end(), decltype(exp_input._vals)::value_type(0));
        sp::Matrix2D<float> result(input._cols, input._rows);
        for(int i=0; i<input._vals.size(); i++){
            result._vals[i] = exp_input._vals[i] / sum_of_exp;
        }
        return result;
    }

    inline Matrix2D<float> D_Softmax(sp::Matrix2D<float> input){ //input:(1row, 3col)
        Matrix2D<float> tensor_1 (input._cols, input._cols);//3*3
        for(int k=0; k<input._cols; k++){
            for(int i=0; i<input._cols; i++){
                tensor_1.at(k,i) = input._vals[k];
            }
        }

        sp::Matrix2D<float> tensor_1_T = tensor_1.transpose();
        tensor_1 = tensor_1.multiplyElements(tensor_1_T);

        Matrix2D<float> tensor_2 (input._cols, input._cols);
        std::fill_n(tensor_2._vals.begin(), input._cols * input._rows, 0);
        for (int i=0; i<input._vals.size(); i++){
            tensor_2.at(i,i) = input._vals[i]; // 对角填充
        }
        Matrix2D<float> tensor_1_nega = tensor_1.negetive();
        Matrix2D<float> result = tensor_2.add(tensor_1_nega); // tsr2-tsr1
        return result; //3*3

    }

    inline sp::Matrix2D<float>Attention(sp::Matrix2D<float> input)
    {
        sp::Matrix2D<float> q(input); //copy input to q
        sp::Matrix2D<float> k(input); //copy input to k
        sp::Matrix2D<float> v(input); //copy input to v

        sp::Matrix2D<float> atten = q.transpose().multiply(k); // atten = k * qT
        atten = atten.multiplyScaler(0.125); // atten * 1/8
        std::cout<< "q*k is ok..."<<std::endl;
        //sp::Matrix2D<float> atten_sig = atten.applyFunction(Sigmoid); // square: attention score
        sp::Matrix2D<float> atten_sig = Softmax(atten); // square: attention score, (301,301)
        std::cout<< "softmax(q*k) is ok..."<<std::endl;
        sp::Matrix2D<float> attention = v.multiply(atten_sig); // same shape as input, (301,64)
        std::cout<< "softmax(q*k)*v is ok..."<<std::endl;

        return attention;
    }

    inline sp::Matrix2D<float> Layer_Norm (sp::Matrix2D<float> input, sp::Matrix2D<float> weight, sp::Matrix2D<float> bias)
    {
        assert(input._cols == weight._rows && input._cols == bias._rows);
        sp::Matrix2D<float> result (input._cols, input._rows); // norm in each cols
        //weight: (64,1),    bias:(64,1)
        float epsln = 1e-05;

        for (int col_id=0; col_id<input._cols; col_id++){ // for each col
            float col_sum = 0.0;
            for (int row_id=0; row_id<input._rows; row_id++){ //sum one col elements
                col_sum += input.at(col_id, row_id);
            }
            float col_mean = col_sum/input._rows; //mean
            float accum = 0.0;
            for (int row_id=0; row_id<input._rows; row_id++){
                accum += (input.at(col_id, row_id) - col_mean) * (input.at(col_id, row_id) - col_mean); //for each row in one col, (x-mul)^2
            }
            float var = accum / input._rows;

            for (int row_id=0; row_id<result._rows; row_id++){
                result.at(col_id, row_id) = (input.at(col_id, row_id)-col_mean)*weight.at(0,col_id)
                        / sqrt(var+epsln) + bias.at(0,col_id);
            }

        }
        return result;

    }

    inline sp::Matrix2D<float> celoss(sp::Matrix2D<float> pred, sp::Matrix2D<float> target)
    {
        sp::Matrix2D<float> log_pred(pred._cols, 1);
        for(int i=0; i<pred._vals.size(); i++){ // log the pred
            log_pred._vals[i] = std::log(pred._vals[i]);
        }
        sp::Matrix2D<float> error = log_pred.multiplyElements(target);
        error = error.negetive();
        return error;
    }

    inline sp::Matrix2D<float> Cat (sp::Matrix2D<float> input_1, sp::Matrix2D<float> input_2){
        sp::Matrix2D<float> result(input_1._cols, input_1._rows+input_2._rows);

        for(int row_id=0; row_id<input_1._rows; row_id++){ // fill the 1st matrix
            for(int col_id=0; col_id<input_1._cols; col_id++){
                result.at(col_id, row_id) = input_1.at(col_id, row_id);
            }
        }
        for(int row_id=0; row_id<input_2._rows; row_id++){ // fill the 2nd matrix
            for(int col_id=0; col_id<input_2._cols; col_id++){
                result.at(col_id, row_id+input_1._rows) = input_2.at(col_id, row_id);
            }
        }
        return result;
    }

    inline sp::Matrix2D<float> ReLU (sp::Matrix2D<float> input){
        sp::Matrix2D<float> result (input._cols, input._rows);

        for(int i=0; i<input._vals.size(); i++){
            if (input._vals[i]>0){
                result._vals[i] = input._vals[i];
            }
            else{
                result._vals[i] = 0;
            }
        }
        return result;
    }

    inline std::vector<float> txt_2_vec (std::string filemane){
        std::string x;
        std::ifstream inFile(filemane);
        std::vector<float> result;
        if (!inFile){
            std::cout << "Unable to open file! ";
        }

        while(std::getline(inFile, x, ','))
        {
            result.push_back(atof(x.c_str()));
        }
        return result;
        inFile.close();

    }

// ================================================================================


// ================================================================================
    class Block {
    public:
        std::vector<sp::Matrix2D<float>> _weightMatrices;
        std::vector<sp::Matrix2D<float>> _valueMatrices;
        std::vector<sp::Matrix2D<float>> _biasMatrices;
        std::vector<sp::Matrix2D<float>> _norm_weight;
        std::vector<sp::Matrix2D<float>> _norm_bias;
        std::vector<sp::Matrix2D<float>> _class_out;

        std::vector<int> _topology;

    public:
        Block(std::vector<int> topolo) :
                _topology(topolo),
                _weightMatrices({}),
                _valueMatrices({}),
                _biasMatrices({}),
                _norm_weight({}),
                _norm_bias({}),
                _class_out({})

                {
            for (int i = 0; i < topolo.size() - 1; i++) {
                Matrix2D<float> weightMatrix(topolo[i + 1], topolo[i]); //3
                _weightMatrices.push_back(weightMatrix);

                Matrix2D<float> biasMatrix(1, topolo[i + 1]); //3
                _biasMatrices.push_back(biasMatrix);
            }
            for (int i = 0; i < 2; i++){
                Matrix2D<float> norm(1,64);
                _norm_weight.push_back(norm);
                _norm_bias.push_back(norm);
            }
            _valueMatrices.resize(topolo.size()); //4, num of element, not M shape.
            _class_out.resize(1);
        }

    };


    class SimpleNN {
    public:
        std::vector<Block> _Blocklist;
        float _learningRate;

    public:
        SimpleNN(std::vector<Block> Blocklist, float learningRate = 0.1f) :
                _Blocklist(Blocklist),
                _learningRate(learningRate) {

        }

        bool feedForward(sp::Matrix2D<float> values, int num_blk) {
            sp::Matrix2D<float> class_tok (64, 1);
            sp::Matrix2D<float> cat_class_tok (64, 301);
            sp::Matrix2D<float> posit_enc (64, 301);
            sp::Matrix2D<float> add_posit_enc (64, 301);
            sp::Matrix2D<float> ini_weight (64, 4);
            sp::Matrix2D<float> ini_bias (1, 64);
            sp::Matrix2D<float> atten_input (64, 301);
            sp::Matrix2D<float> atten_out (64, 301);
            sp::Matrix2D<float> normed (64, 301);

            //===========================================
            //                                          =
            //           Fill values Out of BLOCKs      =
            //                                          =
            //===========================================
            std::string file;
            file = "..\\param\\cls_token.txt";
            std::vector<float> cls_token = txt_2_vec(file); // 64

            file = "..\\param\\pos_embedding.txt";
            std::vector<float> pos_embedding = txt_2_vec(file); // 19264

            file = "..\\param\\init_weight.txt";
            std::vector<float> init_w = txt_2_vec(file);

            file = "..\\param\\init_bias.txt";
            std::vector<float> init_b = txt_2_vec(file);

            class_tok = class_tok.fill_value(cls_token);
            posit_enc = posit_enc.fill_value(pos_embedding);
            ini_weight = ini_weight.fill_value(init_w);
            ini_bias = ini_bias.fill_value(init_b);


            // ini weight, bias     input(300, 4) * w(4, 64) -> (300, 64)
            values = values.multiply(ini_weight);
            values = values.addBias(ini_bias);
            std::cout<<" input success... "<<std::endl;

            //-----cls_tok----------
            cat_class_tok = Cat(class_tok, values);
            values = cat_class_tok; // (301, 64)
            std::cout<<" cat cls_tok finished..."<<std::endl;

            //-----add position enc---------
            add_posit_enc = values.add(posit_enc); //301,64
            std::cout<<" add_posit_enc.shape, "<<add_posit_enc._rows<<", "<<add_posit_enc._cols<<std::endl;
            values = add_posit_enc;


            //-----Attention Block----------
            for (int blk = 0; blk < num_blk; blk++) { // 0 1 2...
                //=================================================
                //
                //      Write in params of BLOCK[blk]:
                //      norm_w0, norm_b0, norm_w1, norm_b1
                //      w0, w1, b1, w2, b2
                //
                //=================================================

                // .txt to vector=======
                std::string file;
                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_norm_w0.txt";
                std::vector<float> blk0_norm_w0 = txt_2_vec(file);

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_norm_b0.txt";
                std::vector<float> blk0_norm_b0 = txt_2_vec(file);

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_norm_w1.txt";
                std::vector<float> blk0_norm_w1 = txt_2_vec(file);

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_norm_b1.txt";
                std::vector<float> blk0_norm_b1 = txt_2_vec(file);

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                                    + std::to_string(blk) + "_w0.txt";
                std::vector<float> blk0_w0 = txt_2_vec(file);
                std::cout<<blk0_w0.size()<<std::endl;
                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_w1.txt";
                std::vector<float> blk0_w1 = txt_2_vec(file);
                std::cout<<blk0_w1.size()<<std::endl;

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_b1.txt";
                std::vector<float> blk0_b1 = txt_2_vec(file);
                std::cout<<blk0_b1.size()<<std::endl;

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_w2.txt";
                std::vector<float> blk0_w2 = txt_2_vec(file);
                std::cout<<blk0_w2.size()<<std::endl;

                file = "..\\param\\BLK " + std::to_string(blk) + "\\blk"
                       + std::to_string(blk) + "_b2.txt";
                std::vector<float> blk0_b2 = txt_2_vec(file);
                std::cout<<blk0_b2.size()<<std::endl;
                // fill in vec -> block_param_list
                _Blocklist[blk]._norm_weight[0] = _Blocklist[blk]._norm_weight[0].fill_value(blk0_norm_w0);
                _Blocklist[blk]._norm_bias[0] = _Blocklist[blk]._norm_bias[0].fill_value(blk0_norm_b0);
                _Blocklist[blk]._norm_weight[1] = _Blocklist[blk]._norm_weight[1].fill_value(blk0_norm_w1);
                _Blocklist[blk]._norm_bias[1] = _Blocklist[blk]._norm_bias[1].fill_value(blk0_norm_b1);

                _Blocklist[blk]._weightMatrices[0] = _Blocklist[blk]._weightMatrices[0].fill_value(blk0_w0);
                _Blocklist[blk]._weightMatrices[1] = _Blocklist[blk]._weightMatrices[1].fill_value(blk0_w1);

                _Blocklist[blk]._biasMatrices[1] = _Blocklist[blk]._biasMatrices[1].fill_value(blk0_b1);
                _Blocklist[blk]._weightMatrices[2] = _Blocklist[blk]._weightMatrices[2].fill_value(blk0_w2);
                _Blocklist[blk]._biasMatrices[2] = _Blocklist[blk]._biasMatrices[2].fill_value(blk0_b2);


                //=================================================
                //=================================================

                _Blocklist[blk]._valueMatrices[0] = values; //(301, 64)
//                for (auto value: values._vals)
//                    std::cout <<"attention input: "<<value << std::endl;
                //-----LayerNorm----------
                values = Layer_Norm(values, _Blocklist[blk]._norm_weight[0], _Blocklist[blk]._norm_bias[0]);
                std::cout<<" LayerNorm finished..."<<std::endl;

                //-----Attention-----------
                atten_input = values; //(301, 64)
                values = atten_input.multiply(_Blocklist[blk]._weightMatrices[0]); // (301, 64)*(64, 64)->(301, 64)
                _Blocklist[blk]._valueMatrices[1] = values; //(301, 64)
                atten_out = Attention(values); //(301, 64)
                values = atten_out;
                std::cout<<" Attention ok..."<<std::endl; std::cout<<" atten_input.shape, "<<atten_input._rows<<", "<<atten_input._cols<<std::endl; std::cout<<" atten_output.shape, "<<atten_out._rows<<", "<<atten_out._cols<<std::endl;
                values = values.add(atten_input); //resident
                std::cout<<" resident ok..."<<std::endl;

                normed = Layer_Norm(values, _Blocklist[blk]._norm_weight[1], _Blocklist[blk]._norm_bias[1]);
                values = normed;

                //-----MLP Layers----------
                values = values.multiply(_Blocklist[blk]._weightMatrices[1]); //(301, 64)*(64, 256)->(301, 256)
                values = values.addBias(_Blocklist[blk]._biasMatrices[1]); //bias(256), is shape of(out_features)
                _Blocklist[blk]._valueMatrices[2] = values;
                values = ReLU(values);
                values = values.multiply(_Blocklist[blk]._weightMatrices[2]); //(301, 256)*(256, 64)->(301, 64)
                values = values.addBias(_Blocklist[blk]._biasMatrices[2]); //bias(64)
                _Blocklist[blk]._valueMatrices[3] = values;
                std::cout<<" MLP Layers finished..."<<std::endl;

                values = values.add(normed); // resident (301, 64)
                std::cout<<"Block: "<<blk<<" finished======================"<<std::endl;

            }

            //-----take 1st value as class----------
            sp::Matrix2D<float> class_token (64, 1);
            sp::Matrix2D<float> final_w (5, 64);
            sp::Matrix2D<float> final_b (5,1);
            sp::Matrix2D<float> class_out (5,1);

            file = "..\\param\\final_weight.txt";
            std::vector<float> final_weight = txt_2_vec(file);
            file = "..\\param\\final_bias.txt";
            std::vector<float> final_bias = txt_2_vec(file);
            final_w = final_w.fill_value(final_weight);
            final_b = final_b.fill_value(final_bias);

            for (int col_id=0; col_id<values._cols; col_id++){
                class_token.at(col_id, 0) = values.at(col_id, 0);
            }
            class_out = class_token.multiply(final_w);
            class_out = class_out.add(final_b);
            std::cout<<" class_out.shape, "<<class_out._rows<<", "<<class_out._cols<<std::endl;
            _Blocklist.back()._class_out[0] = class_out;


            // forward test *************************************
//            for (int blk = 0; blk < num_blk; blk++){
//
//                for (int i = 0; i < _Blocklist[blk]._weightMatrices.size(); i++){ // 1 0
//                    for (auto value: _Blocklist[blk]._weightMatrices[i]._vals)
//                        std::cout <<"b"<<blk<<"w"<<i<<": "   <<value << std::endl;
//                    for (auto value: _Blocklist[blk]._valueMatrices[i]._vals)
//                        std::cout <<"b"<<blk<<"v"<<i<<": "   <<value << std::endl;
//                }
//                for (auto value: _Blocklist[blk]._valueMatrices.back()._vals)
//                    std::cout <<"b"<<blk<<"v"<<_Blocklist[blk]._valueMatrices.size()-1<<": " <<value << std::endl;
//            }

            return true;
//
        }

            std::vector<float> getPredictions() {
//                return _Blocklist.back()._valueMatrices.back()._vals;
                return _Blocklist.back()._class_out[0]._vals;
            }

        };
    }