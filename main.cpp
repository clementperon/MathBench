#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/norm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <vectorial/vec3f.h>
#include <vectorial/vec4f.h>
#include <vectorial/mat4f.h>

// OpenCV uses row-major
// GLM uses column-major
// Eigen uses column-major

// Define a template struct for type mapping
template <typename T>
struct EigenEquivalent;

template <>
struct EigenEquivalent<glm::dmat4>
{
        using type = Eigen::Matrix4d;
};

template <>
struct EigenEquivalent<cv::Matx44d>
{
        using type = Eigen::Matrix4d;
};

template <>
struct EigenEquivalent<glm::fmat4>
{
        using type = Eigen::Matrix4f;
};

template <>
struct EigenEquivalent<cv::Matx44f>
{
        using type = Eigen::Matrix4f;
};

template <>
struct EigenEquivalent<vectorial::mat4f>
{
        using type = Eigen::Matrix4f;
};

template <>
struct EigenEquivalent<glm::dmat3>
{
        using type = Eigen::Matrix3d;
};

template <>
struct EigenEquivalent<cv::Matx33d>
{
        using type = Eigen::Matrix3d;
};

template <>
struct EigenEquivalent<glm::fmat3>
{
        using type = Eigen::Matrix3f;
};

template <>
struct EigenEquivalent<cv::Matx33f>
{
        using type = Eigen::Matrix3f;
};

template <>
struct EigenEquivalent<glm::dvec3>
{
        using type = Eigen::Vector3d;
};

template <>
struct EigenEquivalent<glm::fvec3>
{
        using type = Eigen::Vector3f;
};

template <>
struct EigenEquivalent<vectorial::vec3f>
{
        using type = Eigen::Vector3f;
};

template <>
struct EigenEquivalent<glm::dvec4>
{
        using type = Eigen::Vector4d;
};

template <>
struct EigenEquivalent<glm::fvec4>
{
        using type = Eigen::Vector4f;
};

template <>
struct EigenEquivalent<vectorial::vec4f>
{
        using type = Eigen::Vector4f;
};

template <>
struct EigenEquivalent<glm::fquat>
{
        using type = Eigen::Quaternionf;
};

// Define functors for Operation and Operation
struct Multiply
{
        template <typename T>
        T operator()(const T &a, const T &b) const { return a * b; }
};

struct Add
{
        template <typename T>
        T operator()(const T &a, const T &b) const { return a + b; }
};

struct Normalize
{
        template <typename T>
        auto operator()(T &a, T &b)
        {
                if constexpr (std::is_same<T, Eigen::Vector4f>::value)
                        return a.norm();
                else if constexpr (std::is_same<T, glm::fvec4>::value)
                        return glm::normalize(a);
                else if constexpr (std::is_same<T, vectorial::vec4f>::value)
                        return normalize(a);
        }
};

struct Distance2
{
        template <typename T>
        auto operator()(T &a, T &b)
        {
                if constexpr (std::is_same<T, Eigen::Vector4f>::value)
                        return (b - a).squaredNorm();
                else if constexpr (std::is_same<T, glm::fvec4>::value)
                        return glm::distance2(a, b);
                else if constexpr (std::is_same<T, vectorial::vec4f>::value)
                        return vectorial::length_squared(b - a);
        }
};

// Generic operate function template
template <typename Operation, typename T>
auto operate(T &a, T &b, Operation op)
{
        return op(a, b);
}

// Eigen Matrix Operation
template <typename MatrixType, typename Operation>
static void BM_EigenMatrixOperation(benchmark::State &state)
{
        MatrixType a = MatrixType::Random();
        MatrixType b = MatrixType::Random();

        for (auto _ : state)
        {
                auto result = operate(a, b, Operation{}).eval();
                benchmark::DoNotOptimize(result);
                benchmark::ClobberMemory();
        }
}

// GLM Matrix Operation
template <typename GLMMatrixType, typename EigenMatrixType>
void convertGLMMatrix(const EigenMatrixType &matrix, GLMMatrixType &glmMatrix)
{
        // Eigen and GLM are both column-major
        for (int i = 0; i < matrix.cols(); ++i)
                for (int j = 0; j < matrix.rows(); ++j)
                        glmMatrix[i][j] = matrix(i, j);
}

template <typename MatrixType, typename Operation>
static void BM_GLMMatrixOperation(benchmark::State &state)
{
        MatrixType a;
        MatrixType b;

        using EigenType = typename EigenEquivalent<MatrixType>::type;

        // Use Eigen as reference for Random
        convertGLMMatrix(EigenType::Random(), a);
        convertGLMMatrix(EigenType::Random(), b);

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// OpenCV Matrix Operation
template <typename OpenCVMatrixType, typename EigenMatrixType>
void convertOpenCVMatrix(const EigenMatrixType &matrix, OpenCVMatrixType &cv_matrix)
{
        // Eigen is column-major and OpenCV is row-major
        for (int i = 0; i < matrix.cols(); ++i)
                for (int j = 0; j < matrix.cols(); ++j)
                        cv_matrix(j, i) = matrix(i, j);
}

template <typename MatrixType, typename Operation>
static void BM_OpenCVMatrixOperation(benchmark::State &state)
{
        MatrixType a;
        MatrixType b;

        using EigenType = typename EigenEquivalent<MatrixType>::type;

        convertOpenCVMatrix(EigenType::Random(), a);
        convertOpenCVMatrix(EigenType::Random(), b);

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// Eigen Vector Operation
template <typename VectorType, typename Operation>
static void BM_EigenVectorOperation(benchmark::State &state)
{
        VectorType a = VectorType::Random();
        VectorType b = VectorType::Random();

        for (auto _ : state)
        {
                auto result = (a + b).eval();
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

template <typename GLMVectorType, typename EigenVectorType>
void convertGLMVector(const EigenVectorType &eigen_vector, GLMVectorType &vector)
{
        for (int i = 0; i < eigen_vector.size(); ++i)
                vector[i] = eigen_vector(i);
}

// GLM Vector Operation
template <typename VectorType, typename Operation>
static void BM_GLMVectorOperation(benchmark::State &state)
{
        VectorType a;
        VectorType b;

        using EigenType = typename EigenEquivalent<VectorType>::type;

        convertGLMVector(EigenType::Random(), a);
        convertGLMVector(EigenType::Random(), b);

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// Vectorial Vector Operation
template <typename VectorialVectorType, typename EigenVectorType>
void convertVectorialVector(const EigenVectorType &eigen_vector, VectorialVectorType &vector)
{
        if constexpr (std::is_same<EigenVectorType, Eigen::Vector3f>::value)
        {
                vector = VectorialVectorType(eigen_vector(0), eigen_vector(1), eigen_vector(2));
        }
        else if constexpr (std::is_same<EigenVectorType, Eigen::Vector4f>::value)
        {
                vector = VectorialVectorType(eigen_vector(0), eigen_vector(1), eigen_vector(2), eigen_vector(3));
        }
        else
        {
                assert(false);
        }
}

template <typename VectorType, typename Operation>
static void BM_VectorialVectorOperation(benchmark::State &state)
{
        VectorType a;
        VectorType b;

        using EigenType = typename EigenEquivalent<VectorType>::type;

        // Use Eigen as reference for Random
        convertVectorialVector(EigenType::Random(), a);
        convertVectorialVector(EigenType::Random(), b);

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// Eigen Quaternion Operation
template <typename QuaternionType, typename Operation>
static void BM_EigenQuaternionOperation(benchmark::State &state)
{
        QuaternionType a(QuaternionType::UnitRandom());
        QuaternionType b(QuaternionType::UnitRandom());

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// GLM Quaternion Operation
template <typename GLMQuaternionType, typename EigenQuaternionType>
void convertGLMQuaternion(const EigenQuaternionType &eigen_quat, GLMQuaternionType &quat)
{
        if constexpr (std::is_same<EigenQuaternionType, Eigen::Vector3f>::value)
        {
                quat = GLMQuaternionType(eigen_quat(0), eigen_quat(1), eigen_quat(2));
        }
        else if constexpr (std::is_same<EigenQuaternionType, Eigen::Vector4f>::value)
        {
                quat = GLMQuaternionType(eigen_quat(0), eigen_quat(1), eigen_quat(2), eigen_quat(3));
        }
        else
        {
                assert(false);
        }
}

template <typename QuaternionType, typename Operation>
static void BM_GLMQuaternionOperation(benchmark::State &state)
{
        QuaternionType a;
        QuaternionType b;

        using EigenType = typename EigenEquivalent<QuaternionType>::type;

        convertGLMQuaternion(EigenType::UnitRandom(), a);
        convertGLMQuaternion(EigenType::UnitRandom(), b);

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(operate(a, b, Operation{}));
                benchmark::ClobberMemory();
        }
}

// Quaternion
BENCHMARK_TEMPLATE(BM_EigenQuaternionOperation, Eigen::Quaternionf, Multiply);
BENCHMARK_TEMPLATE(BM_GLMQuaternionOperation, glm::fquat, Multiply);

// Matrix
BENCHMARK_TEMPLATE(BM_EigenMatrixOperation, Eigen::Matrix4f, Multiply);
BENCHMARK_TEMPLATE(BM_GLMMatrixOperation, glm::fmat4, Multiply);
BENCHMARK_TEMPLATE(BM_OpenCVMatrixOperation, cv::Matx44f, Multiply);
BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::mat4f, Multiply);

BENCHMARK_TEMPLATE(BM_EigenMatrixOperation, Eigen::Matrix4f, Add);
BENCHMARK_TEMPLATE(BM_GLMMatrixOperation, glm::fmat4, Add);
BENCHMARK_TEMPLATE(BM_OpenCVMatrixOperation, cv::Matx44f, Add);
// BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::mat4f, Add);

// BENCHMARK_TEMPLATE(BM_EigenMatrixOperation, Eigen::Matrix3d, Multiply);
// BENCHMARK_TEMPLATE(BM_GLMMatrixOperation, glm::dmat3, Multiply);
// BENCHMARK_TEMPLATE(BM_OpenCVMatrixOperation, cv::Matx33d, Multiply);

BENCHMARK_TEMPLATE(BM_EigenMatrixOperation, Eigen::Matrix3f, Multiply);
BENCHMARK_TEMPLATE(BM_GLMMatrixOperation, glm::fmat3, Multiply);
BENCHMARK_TEMPLATE(BM_OpenCVMatrixOperation, cv::Matx33f, Multiply);

BENCHMARK_TEMPLATE(BM_EigenMatrixOperation, Eigen::Matrix3f, Add);
BENCHMARK_TEMPLATE(BM_GLMMatrixOperation, glm::fmat3, Add);
BENCHMARK_TEMPLATE(BM_OpenCVMatrixOperation, cv::Matx33f, Add);

// BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector4d, Multiply);
// BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::dvec4, Multiply);

BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector4f, Add);
BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::fvec4, Add);
BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::vec4f, Add);

BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector4f, Normalize);
BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::fvec4, Normalize);
BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::vec4f, Normalize);

BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector4f, Distance2);
BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::fvec4, Distance2);
BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::vec4f, Distance2);

// BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector3d, Multiply);
// BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::dvec3, Multiply);

BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector3f, Add);
BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::fvec3, Add);
BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::vec3f, Add);

// BENCHMARK_TEMPLATE(BM_EigenVectorOperation, Eigen::Vector3f, Multiply);
// BENCHMARK_TEMPLATE(BM_GLMVectorOperation, glm::fvec3, Multiply);
// BENCHMARK_TEMPLATE(BM_VectorialVectorOperation, vectorial::vec3f, Multiply);

BENCHMARK_MAIN();
