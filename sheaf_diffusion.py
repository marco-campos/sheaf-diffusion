import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from scipy.linalg import expm, null_space

class SheafDiffusion:
    def __init__(self, graph_structure, node_spaces, edge_spaces, connecting_maps):
        """
        Initializes the SheafDiffusion class and computes the delta and sheaf Laplacian matrices.

        Parameters:
        - graph_structure: Dictionary defining the graph structure with edge labels and node connectivity.
        - node_spaces: Dictionary defining dimensions of the vector spaces for each node.
        - edge_spaces: Dictionary defining dimensions of the vector spaces for each edge.
        - connecting_maps: Dictionary defining linear maps for each edge, specified in tuples (source map, target map).
        """
        # Save the graph information
        self.graph_structure = graph_structure
        self.node_spaces = node_spaces
        self.edge_spaces = edge_spaces
        self.connecting_maps = connecting_maps

        # Construct the delta matrix
        self.delta_matrix, self.orientations = couboundary_matrix(graph_structure, node_spaces, edge_spaces, connecting_maps)

        # Compute the sheaf Laplacian: L_F = delta^T * delta
        self.sheaf_laplacian = np.matmul(self.delta_matrix.T, self.delta_matrix)

        # Compute the kernel of the sheaf Laplacian
        self.kernel = null_space(self.sheaf_laplacian)

    def check_initial_condition(self, initial_condition):
        """
        Checks if the initial condition is compatible with the node spaces.

        Parameters:
        - initial_condition: List of numpy arrays, one for each node in the graph.

        Returns:
        - True if dimensions match, otherwise raises an error.
        """
        if len(initial_condition) != len(self.node_spaces):
            raise ValueError("Number of initial conditions does not match number of nodes.")
        
        # Check dimensions for each node space
        for node, vec in zip(self.node_spaces.keys(), initial_condition):
            expected_dim = self.node_spaces[node]
            if vec.shape[0] != expected_dim:
                raise ValueError(f"Initial condition for node {node} has dimension {vec.shape[0]} "
                                 f"but expected dimension is {expected_dim}.")
        return True

    def concatenate_node_vectors(self, initial_condition):
        """
        Concatenates the node vectors into a single vector for the diffusion process.

        Parameters:
        - initial_condition: List of numpy arrays, one for each node in the graph.

        Returns:
        - A single concatenated numpy array representing the initial condition vector x.
        """
        return np.concatenate(initial_condition)

    def run_diffusion(self, initial_condition, alpha, dt, num_timesteps):
        """
        Runs the diffusion process using the equation: dx/dt = -alpha * L_F * x

        Parameters:
        - initial_condition: List of numpy arrays representing the initial vector at t = 0.
        - alpha: Diffusion coefficient (must be > 0).
        - dt: Time step size for explicit time-stepping.
        - num_timesteps: Number of timesteps for the simulation.

        Returns:
        - x: The state vector at the final timestep.
        - all_states: List of state vectors at each timestep.
        """
        # Check and set the initial condition
        self.check_initial_condition(initial_condition)
        x = self.concatenate_node_vectors(initial_condition)

        # Store the initial state
        all_states = [x.copy()]

        # Time-stepping loop
        for _ in range(num_timesteps):
            # Update x using the diffusion equation: x_new = x - alpha * dt * L_F * x
            x = x - alpha * dt * np.matmul(self.sheaf_laplacian, x)
            all_states.append(x.copy())

        return x, all_states

    def analytical_solution(self, initial_condition, alpha, t):
        """
        Computes the analytical solution at time t using the formula: x(t) = exp(-t*alpha*L_F) * x(0)

        Parameters:
        - initial_condition: List of numpy arrays representing the initial vector at t = 0.
        - alpha: Diffusion coefficient.
        - t: Time at which to compute the analytical solution.

        Returns:
        - x_analytical: The analytical solution vector at time t.
        """
        # Set the initial condition vector
        self.check_initial_condition(initial_condition)
        x0 = self.concatenate_node_vectors(initial_condition)

        # Analytical solution: x(t) = exp(-t * alpha * L_F) * x(0)
        exp_matrix = expm(-t * alpha * self.sheaf_laplacian)
        x_analytical = np.dot(exp_matrix, x0)

        return x_analytical

    def compute_projection(self, initial_condition):
        """
        Computes the orthogonal projection of the initial condition onto the kernel of the sheaf Laplacian.

        Parameters:
        - initial_condition: List of numpy arrays representing the initial vector at t = 0.

        Returns:
        - projection: The projection of x(0) onto the kernel of L_F.
        """
        # Check and set the initial condition vector
        self.check_initial_condition(initial_condition)
        x0 = self.concatenate_node_vectors(initial_condition)

        # Orthogonal projection: proj_kernel = Z * Z^T * x0, where Z is the kernel basis
        if self.kernel.size == 0:  # Check if the kernel is trivial
            return np.zeros_like(x0)
        
        projection = np.dot(np.dot(self.kernel, self.kernel.T), x0)
        return projection

class SheafDiffusionVisualizer:
    def __init__(self, sheaf_diffusion, graph_structure, node_positions=None):
        """
        Initializes the visualization class for a given SheafDiffusion instance.

        Parameters:
        - sheaf_diffusion: Instance of SheafDiffusion containing the graph and diffusion dynamics.
        - graph_structure: Graph structure dictionary with edges and their connections.
        - node_positions: Optional dictionary specifying positions of nodes for visualization.
        """
        self.sheaf_diffusion = sheaf_diffusion
        self.graph_structure = graph_structure

        # Create the networkx graph for visualization
        self.G = nx.DiGraph()
        self.G.add_edges_from([(u, v) for u, v in graph_structure.values()])
        
        # If no positions are given, use spring layout
        self.node_positions = node_positions if node_positions else nx.spring_layout(self.G)

    def _node_color_map(self, value, dim):
        """
        Generates a color map for nodes based on their values.
        - For 1D vectors, use grayscale intensity.
        - For 2D vectors, use RGB coloring with the first dimension as red and the second as green.

        Parameters:
        - value: A numpy array representing the node value.
        - dim: Dimension of the vector space at the node.

        Returns:
        - A matplotlib-compatible color for the node.
        """
        norm = Normalize(vmin=-1, vmax=1)

        # Normalize values and ensure they are between 0 and 1
        if dim == 1:
            grayscale_value = np.clip(norm(value[0]), 0, 1)
            return (grayscale_value, grayscale_value, grayscale_value)
        elif dim == 2:
            red = np.clip(norm(value[0]), 0, 1)
            green = np.clip(norm(value[1]), 0, 1)
            return (red, green, 0)

    def plot_diffusion_state(self, state_vector, title="Diffusion State", ax=None):
        """
        Plots the graph at a given state vector.

        Parameters:
        - state_vector: A concatenated vector representing the values of nodes at a given timestep.
        - title: Title of the plot.
        - ax: Matplotlib axis for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Split the state vector back into node values
        node_values = {}
        index = 0
        for node, dim in self.sheaf_diffusion.node_spaces.items():
            node_values[node] = state_vector[index:index + dim]
            index += dim

        # Draw nodes with appropriate coloring
        for node, value in node_values.items():
            color = self._node_color_map(value, len(value))
            nx.draw_networkx_nodes(self.G, self.node_positions, nodelist=[node], node_color=[color],
                                   node_size=500, ax=ax)

        # Draw edges with corresponding colors from nodes
        for edge, (u, v) in self.graph_structure.items():
            # Get values at u and v
            value_u = node_values[u]
            value_v = node_values[v]

            # Map values into the edge space using the corresponding connecting maps
            F_u, F_v = self.sheaf_diffusion.connecting_maps[edge]
            edge_value_u = np.dot(F_u, value_u)
            edge_value_v = np.dot(F_v, value_v)

            # Calculate colors for each half of the edge
            edge_color_u = self._node_color_map(edge_value_u, len(edge_value_u))
            edge_color_v = self._node_color_map(edge_value_v, len(edge_value_v))

            # Draw edge segments for u -> e and v -> e halves
            mid_x = (self.node_positions[u][0] + self.node_positions[v][0]) / 2
            mid_y = (self.node_positions[u][1] + self.node_positions[v][1]) / 2

            # Draw each half with different color
            ax.plot([self.node_positions[u][0], mid_x], [self.node_positions[u][1], mid_y],
                    color=edge_color_u, linewidth=3, linestyle='-')
            ax.plot([mid_x, self.node_positions[v][0]], [mid_y, self.node_positions[v][1]],
                    color=edge_color_v, linewidth=3, linestyle='-')

        # Draw node labels
        nx.draw_networkx_labels(self.G, self.node_positions, ax=ax)
        ax.set_title(title)
        ax.axis('off')

    def animate_diffusion(self, all_states, filename="sheaf_diffusion.mp4", interval=200):
        """
        Animates the diffusion process over time and saves it as a video.

        Parameters:
        - all_states: List of state vectors representing the diffusion at each timestep.
        - filename: Name of the output video file.
        - interval: Time interval between frames in milliseconds.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Initialize the plot with the first state
        def init():
            self.plot_diffusion_state(all_states[0], title="Initial State", ax=ax)
            return ax,

        # Update function for animation
        def update(frame):
            ax.clear()
            self.plot_diffusion_state(all_states[frame], title=f"Timestep: {frame}", ax=ax)
            return ax,

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(all_states), init_func=init,
                                      interval=interval, blit=False, repeat=True)

        # Save the animation as a video file
        ani.save(filename, writer='ffmpeg', dpi=150)
        print(f"Animation saved as {filename}")

def couboundary_matrix(graph_structure, node_spaces, edge_spaces, connecting_maps):
    orientations = {}
    for edge, (node1, node2) in graph_structure.items():
        orientations[edge] = (node1, node2)
    node_offsets = {}
    current_offset = 0
    for node, dim in node_spaces.items():
        node_offsets[node] = current_offset
        current_offset += dim
    edge_offsets = {}
    current_offset = 0
    for edge, dim in edge_spaces.items():
        edge_offsets[edge] = current_offset
        current_offset += dim
    total_node_dim = sum(node_spaces.values())
    total_edge_dim = sum(edge_spaces.values())
    delta = np.zeros((total_edge_dim, total_node_dim))
    for edge, (source, target) in orientations.items():
        source_dim = node_spaces[source]
        target_dim = node_spaces[target]
        edge_dim = edge_spaces[edge]
        F_u_e, F_v_e = connecting_maps[edge]
        edge_start = edge_offsets[edge]
        source_start = node_offsets[source]
        target_start = node_offsets[target]
        if F_u_e.shape != (edge_dim, source_dim):
            raise ValueError(f"Incorrect dimensions for source connecting map of edge {edge}: "
                             f"Expected ({edge_dim}, {source_dim}), got {F_u_e.shape}")
        if F_v_e.shape != (edge_dim, target_dim):
            raise ValueError(f"Incorrect dimensions for target connecting map of edge {edge}: "
                             f"Expected ({edge_dim}, {target_dim}), got {F_v_e.shape}")
        delta[edge_start:edge_start + edge_dim, source_start:source_start + source_dim] = -F_u_e
        delta[edge_start:edge_start + edge_dim, target_start:target_start + target_dim] = F_v_e
    return delta, orientations


if __name__ == "__main__":
    """Example Usage:
    The following example comes from Figure 2 of Ghrist and Hansen's "Opinion Dynamics on Discourse Sheaves"
    """
    graph_structure = {'e1': ('v1', 'v2'), 'e2': ('v2', 'v3'), 'e3': ('v3', 'v4'), 'e4': ('v4', 'v1')}
    node_spaces = {'v1': 1, 'v2': 2, 'v3': 1, 'v4': 2}
    edge_spaces = {'e1': 1, 'e2': 1, 'e3': 1, 'e4': 2}
    connecting_maps = {
        'e1': (np.array([[-2]]), np.array([[-1, 2]])),
        'e2': (np.array([[1, 1]]), np.array([[1]])),
        'e3': (np.array([[-1]]), np.array([[1, -1]])),
        'e4': (np.array([[1, -1], [1, 0]]), np.array([[1], [0]]))
    }

    # Create the sheaf diffusion instance
    sheaf_diffusion = SheafDiffusion(graph_structure, node_spaces, edge_spaces, connecting_maps)

    # Initial condition and parameters
    initial_condition = [np.array([1]), np.array([0, 1]), np.array([0]), np.array([1, -1])]
    alpha = 0.1
    dt = 0.01
    num_timesteps = 1000

    # Run diffusion and compute analytical solution and projection
    numerical_solution, all_states = sheaf_diffusion.run_diffusion(initial_condition, alpha, dt, num_timesteps)
    projection = sheaf_diffusion.compute_projection(initial_condition)


    # Compute and plot the L2 error over time
    l2_errors = []

    # Compute L2 error with the projection
    projection_errors = []
    times = np.arange(0, (num_timesteps + 1) * dt, dt)

    for i, t in enumerate(times):
        l2_projection_error = np.linalg.norm(all_states[i] - projection)
        projection_errors.append(l2_projection_error)
        
        analytical = sheaf_diffusion.analytical_solution(initial_condition, alpha, t)
        l2_error = np.linalg.norm(all_states[i] - analytical)
        l2_errors.append(l2_error)

    # Plot the L2 errors
    plt.plot(times, projection_errors, label='L2 Error (Projection)')
    plt.plot(times, l2_errors, label='L2 Error (Analytical Solution)')

    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title('L2 Error Between Numerical Solution and Projection onto Kernel')
    plt.legend()
    plt.show()
