/** 
 * @author Rakesh Therala
 * @brief A script that solves a 2D poisson equation on a square using finite differences method
 * in parallel using MPI.
 * @version 2.0
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <mpi.h>

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

struct ProcessGrid {
    int rank;           // Current process rank
    int size;           // Total number of processes
    int dims[1];        // Only rows dimension required
    int periods[1];     // Periodicity
    int coords[1];      // Coordinates of the current process in grid
    MPI_Comm comm;      // Cartesian communicator
    int prev_rank;      // Previous rank in d0
    int next_rank;      // Next rank in d0
    int local_nrows;    // Number of local rows
    int start_row;      //Start row index for this process
    int end_row;        //End row index for this process
};

double dirichlet_bc(const double x, const double y)
{
    double g = std::sin(M_PI * x) * std::cos(M_PI * y);
    return g;
}

double rhs_function(const double x, const double y)
{
    double f = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    return f;
}

double analytical_solution(const double x, const double y)
{
    double u = std::sin(M_PI * x) * std::cos(M_PI * y);
    return u;
}

double dot_product(const Vector &vec1, const Vector &vec2,
                            int grid_local_nrows, int int_nodes, int full_col_nodes)
{
    double result = 0.0;
    for (int i = 1; i <= grid_local_nrows; i++) {
        for (int j = 1; j <= int_nodes; j++) {
            int idx = i * full_col_nodes + j;
            result += vec1[idx] * vec2[idx];
        }
    }
    return result;
}

double euclidean_norm(const Vector &vec, int grid_local_nrows, int int_nodes, int full_col_nodes)
{
    return std::sqrt(dot_product(vec, vec, grid_local_nrows, int_nodes, full_col_nodes));
}

void scaled_vector_addition_inplace(Vector &vec1, const Vector &vec2, double alpha, double beta,
                                            int grid_local_nrows, int int_nodes, int full_col_nodes){
        
    for (int i = 1; i <= grid_local_nrows; i++) {
        for (int j = 1; j <= int_nodes; j++) {
            int idx = i * full_col_nodes + j;
            vec1[idx] = alpha * vec1[idx] + beta * vec2[idx];
        }
    }
}

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

void exchange_halo(ProcessGrid &grid,int int_nodes, int full_col_nodes, int full_local_rows, Vector &b_full) 
{
    MPI_Request reqs[4];       // Two sends, two receives

    int req_count = 0;
    if (grid.next_rank != MPI_PROC_NULL) { 
        // Sending last interior row of local process to next rank bottom halo
        MPI_Isend(b_full.data() + ((full_local_rows - 2) * full_col_nodes + 1), int_nodes, MPI_DOUBLE,
                  grid.next_rank, 0, grid.comm, &reqs[req_count++]);
        // Receiving from next rank first interior row into current process top halo
        MPI_Irecv(b_full.data() + ((full_local_rows - 1) * full_col_nodes + 1), int_nodes, MPI_DOUBLE,
                  grid.next_rank, 1, grid.comm, &reqs[req_count++]);
    }
    if (grid.prev_rank != MPI_PROC_NULL) {
        //Sending current process first interior row to previous rank top halo
        MPI_Isend(b_full.data() + (full_col_nodes + 1), int_nodes, MPI_DOUBLE,
                 grid.prev_rank, 1, grid.comm, &reqs[req_count++]);
        //Receiving from previous rank last interior row into current process bottom halo
        MPI_Irecv(b_full.data() + (0 * full_col_nodes + 1), int_nodes, MPI_DOUBLE,
                 grid.prev_rank, 0, grid.comm, &reqs[req_count++]);
    }
    
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

}
void assemble_rhs(const ProcessGrid &grid, int int_nodes, int full_col_nodes, Vector &b_full) {
    // Looping over interior nodes
    double h = 1.0/(int_nodes + 1);
    for (int i = 1; i <= grid.local_nrows; i++) {
        int global_row = i + grid.start_row;  // Global interior row index (ranges from 1 to int_nodes)
        double y = global_row * h;
        for (int j = 1; j <= int_nodes; j++) {
            double x = j * h;
            // Adding rhs function contribution at every interior node
            double rhs_val = rhs_function(x, y);

            // Adding Dirichlet boundary conditions contributions only at the nodes shared with the boundaries nodes
            if (global_row == 1) {
                rhs_val += dirichlet_bc(x, 0.0)/(h * h);    
            }
            if (global_row == int_nodes) {
                rhs_val += dirichlet_bc(x, 1.0)/(h * h);
            }
            if (j == 1) {
                rhs_val += dirichlet_bc(0.0, y)/(h * h);
            }
            if (j == int_nodes) {
                rhs_val += dirichlet_bc(1.0, y)/(h * h);
            }

            //Storing the values in b_full which is fully extended vector contiguous with halo regions
            b_full[i * full_col_nodes + j] = rhs_val;
        }
    }
}

void multiply_Ap(const ProcessGrid &grid, int int_nodes, int full_col_nodes, int full_local_rows, 
    const Vector &b_full, Vector &Ap) 
{

    double h = 1.0 / (int_nodes + 1);

    // Ap is resized to the full extended size.
    Ap.resize(full_local_rows * full_col_nodes, 0.0);

    // Loop over interior nodes
    for (int i = 1; i <= grid.local_nrows; i++) {
        for (int j = 1; j <= int_nodes; j++) {
            // Computing the extended index for the interior cell.
            int idx_center = i * full_col_nodes + j;
            double center = b_full[idx_center];

            // Accessing neighbours in extended array
            double top    = b_full[(i + 1) * full_col_nodes + j];
            double bottom = b_full[(i - 1) * full_col_nodes + j];
            double left   = b_full[i * full_col_nodes + (j - 1)];
            double right  = b_full[i * full_col_nodes + (j + 1)];

            // Compute the 5-point stencil
            double value = (4.0 * center - (top + bottom + left + right)) / (h * h);
            Ap[idx_center] = value;
        }
    }
}

solution CG(const Vector &b_full, double reduction_factor, size_t max_iterations,
            int int_nodes, int full_col_nodes, int full_local_rows, ProcessGrid &grid){

    int interior_size = grid.local_nrows * int_nodes;

    // Allocated vectors (p, r, Ap) of the same size as b_full.
    Vector p_ext = b_full;  // iteration 0: p_ext = b_full,  ext ==> extended vector
    Vector r_ext = b_full;  // initial residual: r_ext = b_full
    Vector Ap_ext(b_full.size(), 0.0);

    Vector x(interior_size, 0.0);   //Solution vector remains at interior size
    
    double local_r_dot = dot_product(r_ext, r_ext, grid.local_nrows, int_nodes, full_col_nodes);
    double global_r_dot;
    MPI_Allreduce(&local_r_dot, &global_r_dot, 1, MPI_DOUBLE, MPI_SUM, grid.comm);
    double initial_residual = std::sqrt(global_r_dot);
    double final_residual = initial_residual;

    size_t iterations = 0;
    double r_dot_old = global_r_dot;

    while (iterations < max_iterations) {

        // First step is to exchange halo regions of p_ext
        exchange_halo(grid, int_nodes, full_col_nodes, full_local_rows, p_ext);
        //Matrix vector multiplication  
        multiply_Ap(grid, int_nodes, full_col_nodes, full_local_rows, p_ext, Ap_ext);
        // Compute p^T * Ap locally
        double local_pAp = dot_product(p_ext, Ap_ext, grid.local_nrows, int_nodes, full_col_nodes);
        double global_pAp;

        MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, grid.comm);
        double alpha = r_dot_old / global_pAp;
        
        // Update solution vector x = x + alpha * p
        for (int i = 1; i <= grid.local_nrows; i++) {
            for (int j = 1; j <= int_nodes; j++) {
                int idx_interior = (i - 1) * int_nodes + (j - 1);      
                int idx_ext = i * full_col_nodes + j;                  
                x[idx_interior] += alpha * p_ext[idx_ext];
            }
        }
    
        // Update interior residual: r = r - alpha * Ap
        scaled_vector_addition_inplace(r_ext, Ap_ext, 1.0, -alpha, grid.local_nrows, int_nodes, full_col_nodes);

        double local_r_dot_new = dot_product(r_ext, r_ext, grid.local_nrows, int_nodes, full_col_nodes);
        double global_r_dot_new;
        MPI_Allreduce(&local_r_dot_new, &global_r_dot_new, 1, MPI_DOUBLE, MPI_SUM, grid.comm);
        double beta = global_r_dot_new / r_dot_old;

        // Update search direction p = r + beta * p
        scaled_vector_addition_inplace(p_ext, r_ext, beta, 1.0, grid.local_nrows, int_nodes, full_col_nodes);
    
        r_dot_old = global_r_dot_new;
        final_residual = std::sqrt(global_r_dot_new);
        iterations++;
        if( final_residual < reduction_factor * initial_residual) break;    //Breaking condition
    
    }
    return { x, iterations, final_residual, initial_residual };

}

int main(int argc, char** argv){
    
    size_t N = 100;
    size_t total_int_nodes = (N-1)*(N-1);
    double h = 1.0/N;
    int max_iterations = 1000;
    double reduction_factor = 1e-6;
    int root =0;

    MPI_Init(&argc, &argv);
    struct ProcessGrid grid;

    grid.dims[0] = 0;   // Let MPI decide the number of rows
    grid.periods[0] = 0; // Non-periodic in row direction

    MPI_Comm_size(MPI_COMM_WORLD, &grid.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &grid.rank);

    // Cartesian topology 1D
    MPI_Dims_create(grid.size, 1, grid.dims); 
    MPI_Cart_create(MPI_COMM_WORLD, 1, grid.dims, grid.periods, 0, &grid.comm);

    // To get coordinates of the current process in the grid
    MPI_Cart_coords(grid.comm, grid.rank, 1, grid.coords);

    // To get previous and next rank in the grid
    MPI_Cart_shift(grid.comm, 0, 1, &grid.prev_rank, &grid.next_rank);

    int int_nodes = N-1;
    int div = int_nodes / grid.dims[0];
    int rem = int_nodes % grid.dims[0];

    //Domain decomposition
    grid.local_nrows = div + (grid.coords[0] < rem ? 1 : 0);
    grid.start_row   = (grid.coords[0] * div) + std::min(grid.coords[0], rem);
    grid.end_row     = grid.start_row + grid.local_nrows;
    
    //Full extended size in each process both columns and rows
    //Full local rows is the number of rows in each process + 2 for halo regions
    //Full column nodes is the number of columns in each process + 2 for halo regions
    int full_col_nodes = int_nodes+2;
    int full_local_rows  =  grid.local_nrows + 2;
    Vector b_full(full_col_nodes*full_local_rows);

    //Vector b_full is the full extended vector with halo regions
    assemble_rhs(grid, int_nodes, full_col_nodes, b_full);  

    double start_time = MPI_Wtime();
    //CG solver
    //The CG function is called with the full extended vector b_full
    //The function will take care of the halo exchange and matrix vector multiplication
    solution sol = CG(b_full, reduction_factor, max_iterations,
        int_nodes, full_col_nodes, full_local_rows, grid);

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    //Gathering the solution vector from all processes
    //The solution vector is gathered in the root process
    int offset =0;
    std::vector<int> rowDivCounts(grid.size,0);
    std::vector<int> displs(grid.size,0);
    std::vector<int> recvCounts(grid.size);

    for (int i = 0; i < grid.size; i++) {
        int proc_rows = div + (i < rem ? 1 : 0);
        recvCounts[i] = proc_rows * int_nodes;
        displs[i] = offset;
        offset += recvCounts[i];
    }

    Vector x(total_int_nodes, 0.0);
    MPI_Gatherv(sol.solution.data(), recvCounts[grid.rank], MPI_DOUBLE,
                x.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                root, grid.comm);

    if (grid.rank == root) {
        double l2_error = discete_l2_norm(N, x);
        std::cout << "The problem size (elements in each direction) is: " << N
                    << " and world size is " << grid.size << "\n";
        std::cout << "CG Solver Results:" << "\n";
        std::cout << "Iterations: " << sol.iterations << "\n";
        std::cout << "Initial Residual: " << sol.initial_residual << "\n";
        std::cout << "Final Residual: " << sol.final_residual << "\n";
        std::cout << "Discrete L2 Error: " << l2_error << "\n";
    }
    std::cout<<"Process: "<< grid.rank<<"\n";
    std::cout << "Time taken(s): " << elapsed << "\n"; 

    MPI_Comm_free(&grid.comm);
    MPI_Finalize();
    return 0;
}