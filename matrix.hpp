#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <functional>


namespace sp
{
    template<typename T>
    class Matrix2D
    {
    public:
        uint32_t _cols;
        uint32_t _rows;
        std::vector<T> _vals;


    public:
        Matrix2D(uint32_t cols, uint32_t rows)
                : _cols(cols),
                  _rows(rows),
                  _vals({})
        {
            _vals.resize(rows * cols, T());
        }

        Matrix2D()
                : _cols(0),
                  _rows(0),
                  _vals({})
        {
        }

        T& at(uint32_t col, uint32_t row)
        {
            return _vals[row * _cols + col];
        }

        bool isSquare()
        {
            return _rows == _cols;
        }


        Matrix2D negetive()
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = -at(x, y);
                }
            return output;
        }

        Matrix2D fill_value(std::vector<float> value_vec)
        {
            Matrix2D output(_cols, _rows);
            for (int i=0; i<output._vals.size(); i++){
                output._vals[i] = value_vec[i];
            }
            return output;
        }


        Matrix2D multiply(Matrix2D& target)
        {
            assert(_cols == target._rows);
            Matrix2D output(target._cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    T result = T();
                    for (uint32_t k = 0; k < _cols; k++)
                        result += at(k, y) * target.at(x, k);
                    output.at(x, y) = result;
                }
            return output;
        }

        Matrix2D multiplyElements(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * target.at(x, y);
                }
            return output;
        }


        Matrix2D add(Matrix2D& target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + target.at(x, y);
                }
            return output;
        }

        Matrix2D addBias(Matrix2D& target)
        {
            assert(_cols == target._rows);
            Matrix2D output(_cols, _rows);
            for (int col_id=0; col_id<output._cols; col_id++){
                for (int row_id=0; row_id<output._rows; row_id++){
                    output.at(col_id, row_id) = at(col_id, row_id) + target.at(0, col_id);
                }
            }
            return output;

        }

        Matrix2D applyFunction(std::function<T(const T&)> func)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = func(at(x, y));
                }
            return output;
        }

        Matrix2D multiplyScaler(float s)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) * s;
                }
            return output;

        }

        Matrix2D addScaler(float s)
        {
            Matrix2D output(_cols, _rows);
            for (uint32_t y = 0; y < output._rows; y++)
                for (uint32_t x = 0; x < output._cols; x++)
                {
                    output.at(x, y) = at(x, y) + s;
                }
            return output;

        }
        Matrix2D transpose()
        {
            Matrix2D output(_rows, _cols);
            for (uint32_t y = 0; y < _rows; y++)
                for (uint32_t x = 0; x < _cols; x++)
                {
                    output.at(y, x) = at(x, y);
                }
            return output;
        }

    }; // class Matrix2D

    template<typename T>
    void LogMatrix2D(Matrix2D<T>& mat)
    {
        for (uint32_t y = 0; y < mat._rows; y++)
        {
            for (uint32_t x = 0; x < mat._cols; x++)
                std::cout << std::setw(10) << mat.at(x, y) << " ";
            std::cout << std::endl;
        }
    }

}
