/**
 * @author Rakesh Therala
 * @brief A script that solves a 2D poisson equation on a square using finite differences method
 * in parallel using OpenMP.
 * @version 1.0
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
typedef std::vector<double> Vector;

struct crs_matrix
{
    std::vector<int> rows;
    std::vector<int> cols;
    Vector vals;
};

struct solution
{
    Vector solution;
    size_t iterations;
    double final_residual, initial_residual;
};

double rhs_function(const double x, const double y)
{
    return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
}

double dirichlet_bc(const double x, const double y)
{
    return std::sin(M_PI * x) * std::cos(M_PI * y);
}

double analytical_solution(const double x, const double y)
{
    return std::sin(M_PI * x) * std::cos(M_PI * y);
}

crs_matrix assemble_FD_stiffness_matrix(const size_t N)
{
    std::vector<int> rows, cols;
    Vector vals;
    int m = N-1;
    double h = 1.0/N;
    double central_elem = 4.0/(h*h);
    double neighbour = -1.0/(h*h);

    rows.push_back(0);

    for(int i=0;i<m;i++){
        for(int j =0; j<m;j++){

            if (i > 0) {  //bottom
                cols.push_back((i - 1) * m + j);
                vals.push_back(neighbour);
            }
            if (j > 0) { //left
                cols.push_back(i * m + j - 1);
                vals.push_back(neighbour);
            }
            //centre
            cols.push_back(i * m + j);
            vals.push_back(central_elem);

            if (j < m-1) { //right
                cols.push_back(i * m + j + 1);
                vals.push_back(neighbour);
            }
           
            if (i < m-1) {  //top
                cols.push_back((i + 1) * m + j);
                vals.push_back(neighbour);
            }
            
            rows.push_back(vals.size());
        }
    }

    return {rows, cols, vals};
}
std::vector<double> assemble_rhs(const size_t N)
{
    int m = N - 1; // Number of internal nodes in each dimension
    int M = m * m; // Total number of internal nodes

    double h = 1.0 / N; // Grid spacing
    Vector b(M, 0.0);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= m; j++) {
            double y = i * h;
            double x = j * h;
            int index = (i - 1) * m + (j - 1);
            b[index] += rhs_function(x, y);

            // Adding boundary contributions
            if (i == 1) {
                b[index] += dirichlet_bc(x,0.0)/(h*h);
            }
            if (i == m) {
                b[index] += dirichlet_bc(x,1.0)/(h*h);
            }
            if (j == 1) {
                b[index] += dirichlet_bc(0.0,y)/(h*h);
            }
            if (j == m) {
                b[index] += dirichlet_bc(1.0,y)/(h*h);
            }
        }
    }
    return b;
}

void sparse_matrix_vector_mult(const crs_matrix &matrix, const std::vector<double> &x, std::vector<double> &result) {
    
    //Parallelizing the loop for sparse matrix vector multiplication
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < x.size(); ++i){
        size_t start_index = matrix.rows[i], end_index = matrix.rows[i + 1];
        result[i] = 0.0;
        for (size_t j = start_index; j < end_index; ++j) {
            result[i] += matrix.vals[j] * x[matrix.cols[j]];
        }
    }
}

double dot_product(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
{
    double result = 0.0;
    //Parallelizing the loop for dot product
    if (vec_1.size() != vec_2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:result)
    #endif
    for (size_t i = 0; i < vec_1.size(); i++) {
        result += vec_1[i] * vec_2[i];
    }
    return result;    
}
double euclidean_norm(const std::vector<double> &x)
{
    double sum = 0.0;
    //Parallelizing the loop for euclidean norm
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum)
    #endif
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }

    return std::sqrt(sum);
}

void scaled_vector_addition_inplace(std::vector<double> &vec_1, const std::vector<double> &vec_2, const double alpha_1, const double alpha_2)
{
    if (vec_1.size() != vec_2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    //Parallelizing the loop for vector addition
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < vec_1.size(); ++i) {
        vec_1[i] = alpha_1 * vec_1[i] + alpha_2 * vec_2[i];
    }
}

double discrete_l2_norm(const size_t N, const std::vector<double> &approx_sol)
{
    int m = N - 1;
    double h = 1.0 / N;

    double error_sum = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction(+:error_sum)
    #endif
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= m; ++j) {
            double y = i * h;
            double x = j * h;

            int index = (i - 1) * m + (j - 1);
            double u_analytical = analytical_solution(x, y);

            double diff = u_analytical - approx_sol[index];
            error_sum += diff * diff;
        }
    }

    double l2_error = std::sqrt(error_sum * h * h);
    return l2_error;
}

solution CG(const crs_matrix &A, const std::vector<double> &b, const double reduction_factor, const size_t max_iterations) {
    size_t M = b.size();
    Vector x(M, 0.0);     // Initial guess x0 = 0
    Vector r = b;         // Initial residual r0 = b - A*x0 = b
    Vector p = r;         // Initial search direction p0 = r0
    Vector Ap(M, 0.0);    // Intializing Ap

    //Computation of A*p before the first iteration starts
    sparse_matrix_vector_mult(A, p, Ap);

    double initial_residual = euclidean_norm(r); // ||r0||
    double final_residual = initial_residual;
    size_t iterations = 0;
    double r_dot_old = dot_product(r, r);       // r0^T * r0

    while (iterations < max_iterations) {

        double alpha = r_dot_old / dot_product(p, Ap);

        scaled_vector_addition_inplace(x, p, 1.0, alpha);

        scaled_vector_addition_inplace(r, Ap, 1.0, -alpha);

        final_residual = euclidean_norm(r);

        double r_dot_new = dot_product(r, r);
        double beta = r_dot_new / r_dot_old;

        scaled_vector_addition_inplace(p, r, beta, 1.0);

        sparse_matrix_vector_mult(A, p, Ap);

        r_dot_old = r_dot_new;
        iterations++;

        if (final_residual < reduction_factor * initial_residual) {
            break;
        }
    }

    return {x, iterations, final_residual, initial_residual};
}

int main(){

    size_t N = 100; //Number of elements in each direction

    double start; 
    double end;
    crs_matrix A = assemble_FD_stiffness_matrix(N);  //CRS matrix assembly
    Vector b = assemble_rhs(N);                     //RHS assembly  
    
    #ifdef _OPENMP 
    start = omp_get_wtime();
    #endif
    double reduction_factor = 1e-6; // Tolerance for convergence
    size_t max_iterations = 1000;   // Max iterations

    #ifdef _OPENMP 
    end = omp_get_wtime();
    #endif

    solution sol = CG(A, b, reduction_factor, max_iterations);

    #ifdef _OPENMP 
    end = omp_get_wtime();
    #endif

    //Results
    std::cout <<"The problem size (elements in each direction) is: "<< N <<std::endl;
    #ifdef _OPENMP
    std::cout<< "The number of threads used is: "<< omp_get_max_threads() <<"\n";
    #endif
    std::cout << "CG Solver Results:" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Initial Residual: " << sol.initial_residual << std::endl;
    std::cout << "Final Residual: " << sol.final_residual << std::endl;

    // Discrete L2 error computation comparing with analytical solution
    double error = discrete_l2_norm(N, sol.solution);
    std::cout << "Discrete L2 Error: " << error << std::endl;

    #ifdef _OPENMP
    std::cout<< "Time taken: "<< end-start<<std::endl;
    #endif
    return 0;
}