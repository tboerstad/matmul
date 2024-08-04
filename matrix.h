#include <vector>

class Matrix {
private:
    std::vector<float> m_data;
    size_t m_rows;
    size_t m_cols;

public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data.resize(rows * cols);
    }

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    float* data() { return m_data.data(); }
    const float* data() const { return m_data.data(); }

    float& operator()(size_t row, size_t col) {
        return m_data[row * m_cols + col];
    }

    const float& operator()(size_t row, size_t col) const {
        return m_data[row * m_cols + col];
    }
};