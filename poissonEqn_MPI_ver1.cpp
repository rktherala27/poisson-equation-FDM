/**
 * @author Rakesh Therala
 * @brief A script that solves a 2D poisson equation on a square using finite differences method
 * in parallel using MPI.
 * @version 1.0
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <mpi.h>

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

            // Add boundary contributions
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
double discete_l2_norm(const size_t N, const std::vector<double> &approx_sol)
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

void sparse_matrix_vector_mult(const crs_matrix &matrix, const std::vector<double> &x, std::vector<double> &result) {
    size_t local_num_rows = matrix.rows.size() - 1;
    for (size_t i = 0; i < local_num_rows; ++i) {
        size_t start_index = matrix.rows[i], end_index = matrix.rows[i + 1];
        result[i] = 0.0;
        for (size_t j = start_index; j < end_index; ++j) {
            result[i] += matrix.vals[j] * x[matrix.cols[j]];
        }
    }
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
solution CG(const crs_matrix &local_A, const std::vector<double> &local_b, 
            const double reduction_factor,
            std::vector<int> recvCounts, std::vector<int> displs, 
            const size_t max_iterations, int global_size, MPI_Comm comm)
{
    int my_rank,world_size;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm,&world_size);

    std::vector<double> p(global_size,0.0);
    std::vector<double> x(global_size,0.0);
    std::vector<double> b(global_size,0.0);
    
    size_t m_local = local_b.size();    // All local initializations
    Vector local_x(m_local, 0.0);    // start guess x0 
    Vector local_r = local_b;        // start residual r = b - A*x0 
    Vector local_p = local_r;      // start search direction
    Vector local_Ap(m_local, 0.0);   // A*p
    
    double local_r_dot = dot_product(local_r, local_r);
    double r_dot_old;
    MPI_Allreduce(&local_r_dot, &r_dot_old, 1, MPI_DOUBLE, MPI_SUM, comm);  // Initial residual norm
    double initial_residual = std::sqrt(r_dot_old);
    double final_residual = initial_residual;
    size_t iterations = 0;         
      
    while (iterations < max_iterations){
        
        //Gathering p full vector in each process
        MPI_Allgatherv(local_p.data(),recvCounts[my_rank], MPI_DOUBLE, p.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, comm);
        sparse_matrix_vector_mult(local_A, p, local_Ap);

        double local_pAp = dot_product(local_p, local_Ap);
        double pAp;
        MPI_Allreduce(&local_pAp, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);

        // Computation of alpha should be consistent in each process
        double alpha = r_dot_old / pAp;

        //local scaled vector addition, no need to gather
        scaled_vector_addition_inplace(local_x, local_p, 1.0, alpha);
        scaled_vector_addition_inplace(local_r, local_Ap, 1.0, -alpha);

        // Computation of new residual squared and beta (consistent in all process)
        double local_r_dot = dot_product(local_r, local_r);
        double r_dot_new;

        //Allreduce for sum of dot products from each process
        MPI_Allreduce(&local_r_dot, &r_dot_new, 1, MPI_DOUBLE, MPI_SUM, comm);
        double beta = r_dot_new / r_dot_old;

        scaled_vector_addition_inplace(local_p, local_r, beta, 1.0);

        // Update for next iteration
        r_dot_old = r_dot_new;

        // Update residual norm and iterations
        final_residual = std::sqrt(r_dot_old);
        iterations++;
        if (final_residual < reduction_factor * initial_residual) break;
        
    }    
    
    return {local_x, iterations, final_residual, initial_residual};
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int root = 0;
    const size_t N = 100;
    const size_t nrows = (N-1)*(N-1); 
    Vector x(nrows,0.0);

    const size_t div = nrows/world_size;
    const size_t rmndr = nrows%world_size;

    int offset =0;
    std::vector<int> rowDivCounts(world_size,0);    //Division of number of rows to each process
    std::vector<int> rowDivRanges(world_size+1,0);  //For process 0, 0 and 1 index gives the range of rows process 0 handles

    std::vector<int> bdispls(world_size,0);     //Displacements vector for rhs scatter
    
    std::vector<int> rowAptrDispls(world_size,0);  
    std::vector<int> rowAptrSendCounts(world_size,0);

    for(int i=0;i<world_size;i++){                  // Making different arrays
        rowDivCounts[i] = div+ (i<rmndr?1:0);       //No. of rows per each process
        rowAptrSendCounts[i] = rowDivCounts[i]+1;   //No. of A.rows to be sent to each process
        rowDivRanges[i] = offset;
        bdispls[i] = offset;                        //Offset added at each iteration for creating displacement vector
        offset += rowDivCounts[i];
    }
    rowDivRanges[world_size] = nrows;               //Adding the last process end rowptr at the last position of array

    std::vector<int> sendCounts(world_size,0);      //SendCounts & displs for scattering values of col_indices and vals
    std::vector<int> displs(world_size,0);

    crs_matrix local_A;
    Vector local_b;
    crs_matrix global_A;
    Vector global_b;

    if(my_rank==root){
        //Assembling A and rhs at root process
        global_A = assemble_FD_stiffness_matrix(N);
        global_b= assemble_rhs(N);
        offset = 0;
        for(int i = 0; i < world_size; ++i) {
            sendCounts[i] = global_A.rows[rowDivRanges[i+1]] - global_A.rows[rowDivRanges[i]];
            displs[i] = offset;
            offset += sendCounts[i];
        }

        local_A.rows.resize(rowAptrSendCounts[my_rank]);
        local_b.resize(rowAptrSendCounts[my_rank]);

        //Scattering row ptrs data to all processes
        MPI_Scatterv(global_A.rows.data(),rowAptrSendCounts.data(), bdispls.data(),MPI_INT,local_A.rows.data(), rowAptrSendCounts[my_rank], MPI_INT, root,MPI_COMM_WORLD);
        
        //Deciding size of column indices and values based on received row ptrs
        int recvCount = local_A.rows[local_A.rows.size()-1] - local_A.rows[0];
        local_A.cols.resize(recvCount);
        local_A.vals.resize(recvCount);

        //Scattering column indices and values to all processes
        MPI_Scatterv(global_A.cols.data(),sendCounts.data(), displs.data(), MPI_INT,local_A.cols.data(), recvCount, MPI_INT, root,MPI_COMM_WORLD);
        MPI_Scatterv(global_A.vals.data(),sendCounts.data(), displs.data(),MPI_DOUBLE,local_A.vals.data(), recvCount, MPI_DOUBLE, root,MPI_COMM_WORLD);
        MPI_Scatterv(global_b.data(),rowDivCounts.data(), bdispls.data(),MPI_DOUBLE,local_b.data(), rowDivCounts[my_rank], MPI_DOUBLE, root,MPI_COMM_WORLD);
    }
    else{
        local_A.rows.resize(rowAptrSendCounts[my_rank]);
        local_b.resize(rowAptrSendCounts[my_rank]);
        
        //Scattering row ptrs data to all processes
        MPI_Scatterv(nullptr,nullptr, nullptr,MPI_INT,local_A.rows.data(), rowAptrSendCounts[my_rank], MPI_INT, root,MPI_COMM_WORLD);
        
        //Deciding size of column indices and values based on received row ptrs
        int recvCount = local_A.rows[local_A.rows.size()-1] - local_A.rows[0];
        local_A.cols.resize(recvCount);
        local_A.vals.resize(recvCount);

        //Scattering column indices and values to all processes
        MPI_Scatterv(nullptr,nullptr, nullptr, MPI_INT,local_A.cols.data(), recvCount, MPI_INT, root,MPI_COMM_WORLD);
        MPI_Scatterv(nullptr,nullptr, nullptr,MPI_DOUBLE,local_A.vals.data(), recvCount, MPI_DOUBLE, root,MPI_COMM_WORLD);
        MPI_Scatterv(nullptr,nullptr, nullptr,MPI_DOUBLE,local_b.data(), rowDivCounts[my_rank], MPI_DOUBLE, root,MPI_COMM_WORLD);
    }

    //Starting the row_ptrs from 0 on each process
    int rowStartValue = local_A.rows[0];
    for(int k=0; k<local_A.rows.size();k++){
        local_A.rows[k] = local_A.rows[k] - rowStartValue;
    }

    double start_time = MPI_Wtime();
    //CG solver in parallel 
    int max_iterations = 1000;
    solution sol = CG(local_A, local_b, 1e-6,rowDivCounts, bdispls, max_iterations, nrows, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    std::vector<double> local_x = sol.solution;

    MPI_Gatherv(local_x.data(), rowDivCounts[my_rank], MPI_DOUBLE,                  //Gathering solution x
                x.data(), rowDivCounts.data(), bdispls.data(), MPI_DOUBLE,
                root, MPI_COMM_WORLD);

    if(my_rank == root) {
        double l2_error = discete_l2_norm(N, x);
        std::cout<<"The problem size (elements in each direction) is: "<< N << "and world size is " << world_size<<"\n";
        std::cout << "CG Solver Results:" << "\n";
        std::cout << "Iterations: " << sol.iterations << "\n";
        std::cout << "Initial Residual: " << sol.initial_residual << "\n";
        std::cout << "Final Residual: " << sol.final_residual << "\n";
        std::cout << "Discrete L2 Error: " << l2_error << "\n";
    } 
    std::cout<<"Process: "<< my_rank<<"\n";
    std::cout << "Time taken(s): " << elapsed << "\n"; 

    std::cout<<"\n";
    
    MPI_Finalize();
    return 0;
}