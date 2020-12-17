// Implement my own mini-version Eigen
// Keyideas:
// (1) resolve all ambiguities at compile time
//     use 'Curiously Recurring Template Pattern' to get 'compile-time' polymorphism
//     use 'traits' and 'forward-declarations' to deduce datatype at compile time
// (2) lazy evaluation, avoid temporary variable (copying twice, one in temporary variable, on from temporary variable to target variable)
//     use 'experssion' eg. CWiseBinaryOp to delay the actual computation util real-usage by calling operator=

#include <iostream>
#include <cassert>
using namespace std;

// ---------------------------------------- Forward Declaration ---------------------------
template <typename T>
struct traits;

template <typename _Scalar, int _Rows, int _Cols>
class Matrix;

template <typename Operation, typename LhsType, typename RhsType>
class CwiseBinaryOp;

enum
{
    Dynamic = -1
};

// ---------------------------------------- Operations ----------------------------------
template <typename _Scalar>
class scalar_sum_op
{
    typedef _Scalar Scalar;

public:
    Scalar operator()(Scalar lhs, Scalar rhs)
    {
        cout << "call scalar_sum_op" << endl;
        return lhs + rhs;
    }

    Scalar operator()(Scalar lhs, Scalar rhs) const
    {
        cout << "call scalar_sum_op" << endl;
        return lhs + rhs;
    }
};

// ---------------------------------------- Datastructure -------------------------------
template <typename _Scalar, int _Rows, int _Cols>
class DenseStorage
{
    typedef _Scalar Scalar;

public:
    DenseStorage()
    {
        if (_Rows != Dynamic && _Cols != Dynamic)
        {
            m_rows = _Rows;
            m_cols = _Cols;
            m_size = m_rows * m_cols;
            array = new Scalar[m_size];
        }
        else
        {
            array = NULL;
        }
    }

    ~DenseStorage()
    {
        if (array != NULL)
            delete[] array;
    }

    Scalar *data() const
    {
        return array;
    }

    int rows() const
    {
        return m_rows;
    }

    int cols() const
    {
        return m_cols;
    }

    int size()
    {
        return m_size;
    }

    int resize(int rows, int cols)
    {
        assert(_Rows == Dynamic && _Cols == Dynamic && "Could not resize fixed-size matrix");
        m_rows = rows;
        m_cols = cols;
        m_size = rows * cols;
        if (array != NULL)
        {
            delete[] array;
        }
        array = new Scalar[m_size];
    }

private:
    int m_size;
    int m_rows;
    int m_cols;
    Scalar *array;
};

template <typename Derived>
class MatrixBase
{
    typedef traits<Derived> MT;
    typedef typename MT::Scalar Scalar;

public:
    MatrixBase()
    {
        cout << "rows: " << MT::RowsAtCompileTime << ", cols: " << MT::ColsAtCompileTime << endl;
    }

    MatrixBase(const MatrixBase<Derived> &other)
    {
        cout << __FILE__ << ", " << __LINE__ << " copy constructor in matrixbase" << endl;
    }

    // ToDO: have not figure out completely yet
    template <typename OtherDerived>
    // CwiseBinaryOp<scalar_sum_op<typename Derived::Scalar>, Derived, OtherDerived> // error! invalid use of incomplete type
    CwiseBinaryOp<scalar_sum_op<typename traits<Derived>::Scalar>, Derived, OtherDerived>
    operator+(const MatrixBase<OtherDerived> &other)
    {
        cout << "perform plus..." << endl;
        scalar_sum_op<typename traits<Derived>::Scalar> op;
        return CwiseBinaryOp<scalar_sum_op<typename traits<Derived>::Scalar>, Derived, OtherDerived>((*this).derived(), other.derived());
    }

    template <typename OtherDerived>
    MatrixBase<Derived> &operator+=(const MatrixBase<OtherDerived> &other)
    {
        for (int i = 0; i < rows(); i++)
        {
            for (int j = 0; j < cols(); j++)
            {
                m_storage.data()[i * cols() + j] += other.m_storage.data()[i * cols() + j];
            }
        }
        return *this;
    }

    template <typename OtherDerived>
    Derived &operator=(const MatrixBase<OtherDerived> &other)
    {
        cout << "calling base operator=" << endl;
        assert(m_storage.data() != 0 && "you cannot use operator= with a non initialized matrix (instead use set()");
        for (int i = 0; i < rows(); i++)
        {
            for (int j = 0; j < cols(); j++)
            {
                m_storage.data()[i * cols() + j] += other.derived()(i, j);
            }
        }
        return (*this).derived();
    }

    int rows() const
    {
        return m_storage.rows();
    }

    int cols() const
    {
        return m_storage.cols();
    }

    Scalar operator()(int i, int j) const
    {
        return m_storage.data()[i * cols() + j];
    }

    bool resize(int rows, int cols)
    {
        m_storage.resize(rows, cols);
    }

    Derived &derived()
    {
        cout << "calling cast..." << endl;
        return *static_cast<Derived *>(this);
    }

    const Derived &derived() const
    {
        cout << "calling const cast..." << endl;
        return *static_cast<const Derived *>(this);
    }

protected:
    DenseStorage<Scalar, MT::RowsAtCompileTime, MT::ColsAtCompileTime> m_storage;
    string name;
};

template <typename Operation, typename LhsType, typename RhsType>
class CwiseBinaryOp : public MatrixBase<CwiseBinaryOp<Operation, LhsType, RhsType>>
{
    typedef typename traits<LhsType>::Scalar Scalar;
    typedef MatrixBase<CwiseBinaryOp<Operation, LhsType, RhsType>> Base;
    using Base::name;

public:
    CwiseBinaryOp(const LhsType &lhs, const RhsType &rhs) : m_lhs(lhs), m_rhs(rhs)
    {
        name = "CwiseOp";
    }

    Scalar operator()(int i, int j)
    {
        // return m_lhs(i, j) + m_rhs(i, j);
        return m_op(m_lhs(i, j), m_rhs(i, j));
    }

    Scalar
    operator()(int i, int j) const
    {
        cout << "CwiseBinaryOp..." << endl;
        // return m_lhs(i, j) + m_rhs(i, j);
        return m_op(m_lhs(i, j), m_rhs(i, j));
    }

private:
    Operation m_op;
    const LhsType &m_lhs;
    const RhsType &m_rhs;
};
template <typename Operation, typename LhsType, typename RhsType>
struct traits<CwiseBinaryOp<Operation, LhsType, RhsType>>
{
    typedef typename traits<LhsType>::Scalar Scalar;
    enum
    {
        RowsAtCompileTime = traits<LhsType>::RowsAtCompileTime,
        ColsAtCompileTime = traits<LhsType>::ColsAtCompileTime
    };
};

template <typename _Scalar, int _Rows, int _Cols>
class Matrix : public MatrixBase<Matrix<_Scalar, _Rows, _Cols>>
{
    typedef MatrixBase<Matrix<_Scalar, _Rows, _Cols>> Base;
    typedef _Scalar Scalar;

    using Base::m_storage;
    using Base::name;

public:
    Matrix() : Base()
    {
        name = "matrix";
    }

    Matrix(const Matrix &other) : Base(other)
    {
        cout << "copy constructor..." << endl;
    }

    Matrix(const Scalar &x, const Scalar &y, const Scalar &z) : Base()
    {
        m_storage.data()[0] = x;
        m_storage.data()[1] = y;
        m_storage.data()[2] = z;
    }

    Matrix &operator=(const Matrix &other)
    {
        cout << __FILE__ << " " << __LINE__ << " Calling matrix operator=..." << endl;
        // return Base::operator=(other);
    }

    template <typename OtherDerived>
    Matrix &operator=(const MatrixBase<OtherDerived> &other)
    {
        cout << __FILE__ << " " << __LINE__ << " Calling matrix operator=..." << endl;
        assert(m_storage.data() != 0 && "you cannot use operator= with a non initialized matrix (instead use set()");
        return Base::operator=(other.derived());
    }
};

template <typename _Scalar, int _Rows, int _Cols>
struct traits<Matrix<_Scalar, _Rows, _Cols>>
{
    typedef _Scalar Scalar;
    enum
    {
        RowsAtCompileTime = _Rows,
        ColsAtCompileTime = _Cols
    };
};

typedef Matrix<float, 3, 1> Vector3f;
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

template <typename Derived>
ostream &operator<<(ostream &out, const MatrixBase<Derived> &mat)
{
    for (int i = 0; i < mat.rows(); i++)
    {
        for (int j = 0; j < mat.cols(); j++)
            out << mat(i, j) << " ";
        cout << endl;
    }
    return out;
}

int main()
{
    Vector3f x(0, 1, 2), y(1, 1, 1), z;
    z = x + y;
    cout << z << endl;

    // MatrixXd X, Y;
    // X.resize(2, 3);
    // Y.resize(2, 3);
    return 0;
}