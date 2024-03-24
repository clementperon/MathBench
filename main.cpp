#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Eigen Matrix Multiplication
static void BM_EigenMatrixMultiplication(benchmark::State &state)
{
        Eigen::Matrix4d a = Eigen::Matrix4d::Random();
        Eigen::Matrix4d b = Eigen::Matrix4d::Random();
        for (auto _ : state)
        {
                auto result = (a * b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenMatrixMultiplication);

// GLM Matrix Multiplication
static void BM_GLMMatrixMultiplication(benchmark::State &state)
{
        glm::mat4 a(1.0f); // Using identity matrices for simplicity
        glm::mat4 b(1.0f);
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a * b);
        }
}
BENCHMARK(BM_GLMMatrixMultiplication);

// Eigen Vector Addition
static void BM_EigenVectorAddition(benchmark::State &state)
{
        Eigen::Vector4d a = Eigen::Vector4d::Random();
        Eigen::Vector4d b = Eigen::Vector4d::Random();
        for (auto _ : state)
        {
                auto result = (a + b).eval();
                benchmark::DoNotOptimize(result);
        }
}
BENCHMARK(BM_EigenVectorAddition);

// GLM Vector Addition
static void BM_GLMVectorAddition(benchmark::State &state)
{
        glm::vec4 a(1.0f);
        glm::vec4 b(1.0f);
        for (auto _ : state)
        {
                benchmark::DoNotOptimize(a + b);
        }
}
BENCHMARK(BM_GLMVectorAddition);

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
