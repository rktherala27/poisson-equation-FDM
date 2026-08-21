/**
 * @author Rakesh Therala
 * @brief A script that solves a 2D poisson equation on a square using finite differences method
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;
typedef std::vector<double> Vector;

struct crs_matrix
{
    std::vector<int> rows;
    std::vector<int> cols;
    Vector vals;
};

/**
 * @brief A structure holding the information about the result from the solver
 */
struct solution
{
    Vector solution;
    size_t iterations;
    double final_residual, initial_residual;
};

std::vector<int> sort(std::vector<int> vec){
    //Vector dummy = unsorted;
    std::sort(vec.begin(),vec.end());
    return vec;
}


double rhs_function(const double x, const double y)
{
    double f = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    return f;
}

double dirichlet_bc(const double x, const double y)
{
    double g = std::sin(M_PI * x) * std::cos(M_PI * y);
    return g;
}

double analytical_solution(const double x, const double y)
{
    double u = std::sin(M_PI * x) * std::cos(M_PI * y);
    return u;
}

/**
 * @brief A function that assembles the stiffness matrix for a system with a uniform grid
 * and same number of nodes in each dimanesion
 *
 * @param N number of elements in one dimension
 * @return crs_matrix
 */
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


/**
 * @brief A function that assembles the right-hand side vector for a system with a uniform grid
 * and same number of nodes in each dimanesion
 *
 * @param N number of elements in one dimension
 * @return std::vector<double>
 */
std::vector<double> assemble_rhs(const size_t N)
{
    int m = N - 1; // Number of internal nodes in each dimension
    int M = m * m; // Total number of internal nodes

    double h = 1.0 / N; // Grid spacing
    Vector b(M, 0.0);

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

/**
 * @brief A function that computes a matrix vector multiplication between a crs matrix and a vector
 *
 * @param matrix
 * @param x
 * @param result
 */   

void sparse_matrix_vector_mult(const crs_matrix &matrix, const std::vector<double> &x, std::vector<double> &result) {
    for (size_t i = 0; i < x.size(); ++i){
        size_t start_index = matrix.rows[i], end_index = matrix.rows[i + 1];
        result[i] = 0.0;
        for (size_t j = start_index; j < end_index; ++j) {
            result[i] += matrix.vals[j] * x[matrix.cols[j]];
        }
    }
}

/**
 * @brief A function that computes the dot product of two vectors
 *
 * @param vec_1
 * @param vec_2
 * @return double
 */
double dot_product(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
{
    double result = 0.0;
    for(int i=0; i<vec_1.size();i++){
        result = result + vec_1[i]*vec_2[i];       
    }
    return result;
}

/**
 * @brief A function that computes the Euclidean norm of a vector
 *
 * @param x
 * @return double
 */
double euclidean_norm(const std::vector<double> &x)
{
    double sum = 0.0;
    for (const auto& val : x) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

/**
 * @brief A function that does an inplace scaled vector addition between two vecotrs
 * vec1 = alpha_1*vec1 + alpha_2*vec2
 *
 * @param vec_1
 * @param vec_2
 * @param alpha_1
 * @param alpha_2
 */
void scaled_vector_addition_inplace(std::vector<double> &vec_1, const std::vector<double> &vec_2, const double alpha_1, const double alpha_2)
{
    if (vec_1.size() != vec_2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    for (size_t i = 0; i < vec_1.size(); i++) {
        vec_1[i] = alpha_1 * vec_1[i] + alpha_2 * vec_2[i];
    }
}
/**
 * @brief A function that compute the discrete L2 norm of the error
 * between the approximate solution and the analytical solution
 *
 * @param N
 * @param approx_sol
 * @return double
 */
double discrete_l2_norm(const size_t N, const std::vector<double> &approx_sol)
{
    int m = N - 1;
    double h = 1.0/N;

    double error_sum = 0.0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= m; ++j) {
            double y= i*h;
            double x = j*h; 

            int index = (i - 1) * m + (j - 1);
            double u_analytical = analytical_solution(x, y);

            double diff = u_analytical - approx_sol[index];
            error_sum += diff * diff;
        }
    }
    double l2_error = std::sqrt(error_sum*h*h);

    return l2_error;
}

/**
 * @brief Sparse CG Solver
 *
 * @param A crs matrix
 * @param b right hand side
 * @param reduction_factor for the stopping condition
 * @param max_iterations
 * @return solution
 */

solution CG(const crs_matrix &A, const std::vector<double> &b, const double reduction_factor, const size_t max_iterations) {
    size_t M = b.size();
    Vector x(M, 0.0);     // Initial guess x0 = 0
    Vector r = b;         // Initial residual r0 = b - A*x0 = b
    Vector p = r;         // Initial search direction p0 = r0
    Vector Ap(M, 0.0);    // Intializing Ap

    //Computation of A*p before the first iteration starts
    sparse_matrix_vector_mult(A, p, Ap); 
    
    for(size_t i = 0; i < M; ++i) {
        std::cout<< Ap[i] << " ";
    }

    double initial_residual = euclidean_norm(r); // ||r0||
    double final_residual = initial_residual;
    size_t iterations = 0;
    double r_dot_old = dot_product(r, r);       // r0^T * r0

    while (iterations < max_iterations) {
        // Alpha computation with old residual and p^T*A*p
        double alpha = r_dot_old / dot_product(p, Ap);

        // Solution Update: x = x + alpha * p
        scaled_vector_addition_inplace(x, p, 1.0, alpha);

        // Residual Update: r = r - alpha * A * p
        scaled_vector_addition_inplace(r, Ap, 1.0, -alpha);

        // Checking convergence by calculating final residual and comparing with initial residual, reduction factor

        // Beta computation: beta = (r_new^T * r_new) / (r_old^T * r_old)
        double r_dot_new = dot_product(r, r);
        double beta = r_dot_new / r_dot_old;

        // Search direction update: p = r + beta * p
        scaled_vector_addition_inplace(p, r, beta, 1.0);

        // Computation of A*p for the next iteration
        sparse_matrix_vector_mult(A, p, Ap);

        // Making current iteration residual to old
        r_dot_old = r_dot_new;
        iterations++;

        final_residual = euclidean_norm(r);
        if (final_residual < reduction_factor * initial_residual) {
            break;
        }
    }

    return {x, iterations, final_residual, initial_residual};
}

int main()
{

    size_t N = 4; //Number of elements in each direction

    crs_matrix A = assemble_FD_stiffness_matrix(N);  //CRS matrix assembly
    Vector b = assemble_rhs(N);                      //RHS assembly

    double reduction_factor = 1e-8; // Tolerance for convergence
    size_t max_iterations = 10000;   // Max iterations

    //Time stamp starts here before solver
    auto start = std::chrono::high_resolution_clock::now();

    solution sol = CG(A, b, reduction_factor, max_iterations);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> solver_time = end - start;

    //Results
    std::cout <<"The problem size (elements in each direction) is: "<< N <<std::endl;
    std::cout << "CG Solver Results:" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Initial Residual: " << sol.initial_residual << std::endl;
    std::cout << "Final Residual: " << sol.final_residual << std::endl;

    // Discrete L2 error computation comparing with analytical solution
    double error = discrete_l2_norm(N, sol.solution);
    std::cout << "Discrete L2 Error: " << error << std::endl;
    std::cout << "Solver Time: " << solver_time.count() << " seconds" << std::endl;

    return 0;
}