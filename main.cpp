#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>

// Eigen Matrix Multiplication
static void BM_EigenMatrix4Multiplication(benchmark::State &state)
{
        Eigen::Matrix4d a = Eigen::Matrix4d::Random();
        Eigen::Matrix4d b = Eigen::Matrix4d::Random();
        for (auto _ : state)
        {
                auto result = (a * b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenMatrix4Multiplication);

static void BM_EigenMatrix3Multiplication(benchmark::State &state)
{
        Eigen::Matrix3d a = Eigen::Matrix3d::Random();
        Eigen::Matrix3d b = Eigen::Matrix3d::Random();
        for (auto _ : state)
        {
                auto result = (a * b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenMatrix3Multiplication);

// GLM Matrix Multiplication
static void BM_GLMMatrix4Multiplication(benchmark::State &state)
{
        Eigen::Matrix4d eigenA = Eigen::Matrix4d::Random();
        Eigen::Matrix4d eigenB = Eigen::Matrix4d::Random();
        glm::mat4 a;
        glm::mat4 b;
        for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row)
                {
                        // Note: glm is column-major
                        a[col][row] = eigenA(row, col);
                        b[col][row] = eigenB(row, col);
                }

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a * b);
        }
}
BENCHMARK(BM_GLMMatrix4Multiplication);

static void BM_GLMMatrix3Multiplication(benchmark::State &state)
{
        Eigen::Matrix3d eigenA = Eigen::Matrix3d::Random();
        Eigen::Matrix3d eigenB = Eigen::Matrix3d::Random();
        glm::mat3 a;
        glm::mat3 b;
        for (int col = 0; col < 3; ++col)
                for (int row = 0; row < 3; ++row)
                {
                        // Note: glm is column-major
                        a[col][row] = eigenA(row, col);
                        b[col][row] = eigenB(row, col);
                }

        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a * b);
        }
}
BENCHMARK(BM_GLMMatrix3Multiplication);

// Eigen Vector Addition
static void BM_EigenVector4Addition(benchmark::State &state)
{
        Eigen::Vector4d a = Eigen::Vector4d::Random();
        Eigen::Vector4d b = Eigen::Vector4d::Random();
        for (auto _ : state)
        {
                auto result = (a + b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenVector4Addition);

static void BM_EigenVector3Addition(benchmark::State &state)
{
        Eigen::Vector3d a = Eigen::Vector3d::Random();
        Eigen::Vector3d b = Eigen::Vector3d::Random();
        for (auto _ : state)
        {
                auto result = (a + b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenVector3Addition);

// GLM Vector Addition
static void BM_GLMVector4Addition(benchmark::State &state)
{
        Eigen::Vector4d eigenA = Eigen::Vector4d::Random();
        Eigen::Vector4d eigenB = Eigen::Vector4d::Random();
        glm::vec4 a(eigenA(0), eigenA(1), eigenA(2), eigenA(3));
        glm::vec4 b(eigenB(0), eigenB(1), eigenB(2), eigenB(3));
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a + b);
        }
}
BENCHMARK(BM_GLMVector4Addition);

static void BM_GLMVector3Addition(benchmark::State &state)
{
        Eigen::Vector3d eigenA = Eigen::Vector3d::Random();
        Eigen::Vector3d eigenB = Eigen::Vector3d::Random();
        glm::vec3 a(eigenA(0), eigenA(1), eigenA(2));
        glm::vec3 b(eigenB(0), eigenB(1), eigenB(2));
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a + b);
        }
}
BENCHMARK(BM_GLMVector3Addition);

// Eigen Quaternion Multiplication
static void BM_EigenQuaternionMultiplication(benchmark::State &state)
{
        Eigen::Quaterniond a(Eigen::Quaterniond::UnitRandom());
        Eigen::Quaterniond b(Eigen::Quaterniond::UnitRandom());
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a * b);
        }
}
BENCHMARK(BM_EigenQuaternionMultiplication);

// GLM Quaternion Multiplication
static void BM_GLMQuaternionMultiplication(benchmark::State &state)
{
        glm::quat a(1.0f, 0.0f, 0.0f, 0.0f);
        glm::quat b(1.0f, 0.0f, 0.0f, 0.0f);
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a * b);
        }
}
BENCHMARK(BM_GLMQuaternionMultiplication);

BENCHMARK_MAIN();
