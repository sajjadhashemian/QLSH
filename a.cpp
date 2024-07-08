#include <bits/stdc++.h>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Helper function to generate truncated normal distribution samples
double sample_truncated_normal(double mean, double stddev, double low, double high)
{
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stddev);

    // double sample;
    // do
    // {
    //     sample = d(gen);
    // } while (sample < low || sample > high);

    return d(gen);
}

int sample_dgauss(double mu, double sigma, double b)
{
    sigma = sigma / b;
    mu = mu / b;

    while (true)
    {
        double candidate = round(sample_truncated_normal(mu, sigma, mu - 5 * b * sigma, mu + 5 * b * sigma));
        double acceptance_prob = exp(-0.5 * pow((candidate - mu) / sigma, 2));
        double u = (double)rand() / RAND_MAX;

        if (u < acceptance_prob)
        {
            return static_cast<int>(candidate);
        }
    }
}

VectorXd sample_dgauss_lattice(const MatrixXd &B, const VectorXd &mu, double sigma)
{
    int n = B.cols();
    VectorXd result(n);

    for (int i = 0; i < n; ++i)
    {
        result(i) = sample_dgauss(mu(i), sigma, B.col(i).norm());
    }

    return B * result;
}

map<vector<int>, vector<VectorXd>> compute_coset_representatives(const MatrixXd &B, const vector<VectorXd> &V)
{
    map<vector<int>, vector<VectorXd>> coset_groups;

    for (const auto &vec : V)
    {
        vector<int> Z(B.cols());
        for (int j = 0; j < B.cols(); ++j)
        {
            Z[j] = static_cast<int>(floor(vec(j) / B.col(j).norm())) % 2;
        }

        Z.back() = (((accumulate(Z.begin(), Z.end(), 0) - Z.back()) % 2) + 2) % 2;
        coset_groups[Z].push_back(vec);
    }

    int counter = 0;
    for (const auto &group : coset_groups)
    {
        if (group.second.size() == 1)
        {
            ++counter;
        }
    }

    if (counter == coset_groups.size())
    {
        return {};
    }

    return coset_groups;
}

vector<VectorXd> pair_and_average(const MatrixXd &B, const vector<VectorXd> &vectors, const map<vector<int>, vector<VectorXd>> &coset_groups)
{
    vector<VectorXd> output;

    for (const auto &group : coset_groups)
    {
        vector<VectorXd> unpaired = group.second;

        while (unpaired.size() > 1)
        {
            VectorXd Xi = unpaired.back();
            unpaired.pop_back();
            VectorXd Xj = unpaired.back();
            unpaired.pop_back();
            output.push_back((Xi + Xj) / 2);
        }
    }

    return output;
}

vector<VectorXd> cCVP(const MatrixXd &L, const VectorXd &t, int p, double sigma)
{
    vector<VectorXd> result(p);
    for (int i = 0; i < p; ++i)
    {
        result[i] = sample_dgauss_lattice(L.transpose(), t, sigma);
    }
    return result;
}

map<vector<int>, VectorXd> group_by_cosets(const vector<VectorXd> &y_set, const MatrixXd &L)
{
    map<vector<int>, VectorXd> coset_groups;

    for (const auto &y : y_set)
    {
        vector<int> Z(L.cols());
        for (int j = 0; j < L.cols(); ++j)
        {
            Z[j] = static_cast<int>(floor(y(j) / L.col(j).norm())) % 2;
        }

        Z.back() = (((accumulate(Z.begin(), Z.end(), 0) - Z.back()) % 2) + 2) % 2;
        coset_groups[Z] = y;
    }

    return coset_groups;
}

VectorXd exact_CVP(const MatrixXd &L, const VectorXd &t, int p, double sigma)
{
    int n = L.rows();
    int m = L.cols();

    if (n == 1)
    {
        double T = floor(t.norm() / L.norm());
        vector<VectorXd> x(10);
        for (int k = -5; k < 5; ++k)
        {
            x[k + 5] = (T + k) * L.col(0);
        }

        auto min_it = min_element(x.begin(), x.end(), [&](const VectorXd &a, const VectorXd &b)
                                  { return (a - t).norm() < (b - t).norm(); });

        return *min_it;
    }

    vector<VectorXd> vectors = cCVP(L, t, p, sigma);

    MatrixXd L_prime = L.block(0, 0, n - 1, m);
    auto coset_groups = group_by_cosets(vectors, L);

    VectorXd closest_point;
    double min_distance = numeric_limits<double>::infinity();

    for (const auto &group : coset_groups)
    {
        VectorXd t_shifted = t.head(n - 1) - group.second.head(n - 1);
        VectorXd x_c = exact_CVP(L_prime, t_shifted, p, sigma);
        VectorXd candidate = x_c + group.second;
        double distance = (candidate - t).norm();

        if (distance < min_distance)
        {
            min_distance = distance;
            closest_point = candidate;
        }
    }

    return closest_point;
}

int main()
{
    vector<int> S = {2, 4, 5};
    int target_sum = 16;
    int Q = 10 * pow(10, round(log10(pow(2, S.size()) * target_sum * *max_element(S.begin(), S.end()))));

    MatrixXd B(S.size() + 1, S.size());
    B.setZero();
    for (size_t i = 0; i < S.size(); ++i)
    {
        B(i, i) = S[i];
    }
    VectorXd tt(S.size());
    for (int i = 0; i < S.size(); i++)
        tt(i) = S[i];
    B.row(S.size()) = Q*tt;

    VectorXd t(S.size() + 1);
    t << VectorXd::Ones(S.size()) * target_sum, Q * target_sum;

    int p = pow(2, S.size() + 5);
    double sigma = 10.0;

    VectorXd closest_vector;
    double min_distance = numeric_limits<double>::infinity();

    for (int i = 0; i < 8; ++i)
    {
        VectorXd candidate = exact_CVP(B.transpose(), t, p, sigma);
        double distance = (candidate - t).norm();

        if (distance < min_distance)
        {
            min_distance = distance;
            closest_vector = candidate;
        }

        cout << i << ": " << candidate.transpose() << endl;
    }

    cout << "Closest vector to target: " << closest_vector.transpose() << endl;

    return 0;
}
