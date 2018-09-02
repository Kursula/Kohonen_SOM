import numpy as np
import sys, math, time
import matplotlib.pyplot as plt


class SOM(object):
    def __init__(self):
        # Nothing here. The map instance content can be created by either 
        # loading from file or running the create method. 
        pass

    def create(self, width, height, ch):
        self.x = width  # Map width
        self.y = height  # Map height
        self.ch = ch # Map channels (vector length)
        self.trained = False

    def save(self, filename):
        # Save SOM to .npy file. 
        if self.trained:
            np.save(filename, self.node_vectors)
            return True
        else:
            return False

    def load(self, filename):
        # Load SOM from .npy file. 
        self.node_vectors = np.load(filename)
        self.x = self.node_vectors.shape[1]
        self.y = self.node_vectors.shape[0]
        self.ch = self.node_vectors.shape[2]
        self.trained = True
        return True

    def get_map_vectors(self):
        # Returns the map vectors. 
        if self.trained:
            return self.node_vectors
        else:
            return False
    
    def distance(self, vect_a, vect_b):
        if self.dist_method == 'euclidean':
            dist = np.linalg.norm(vect_a - vect_b)
        elif self.dist_method == 'cosine':
            dist = 1. - np.dot(vect_a, vect_b) / (np.linalg.norm(vect_a) * np.linalg.norm(vect_b))
        return dist
    
    def find_maching_nodes(self, input_arr):
        # This is to be called only when the map is trained.
        if self.trained == False:
            return False
        
        n_data = input_arr.shape[0]
        locations = np.zeros((n_data, 2), dtype=np.int32)   
        distances = np.zeros((n_data), dtype=np.float32)   
        
        print_step = int(n_data / 20)
        print_count = 0 
        for idx in range(n_data):
        
            if idx % print_step == 0: 
                print_count += 1
                sys.stdout.write(f'\rFinding mathing nodes' +
                 ' [' + '=' * (print_count) + '>' + '.' * (20 - print_count) + '] ')

            data_vect = input_arr[idx]
            min_dist = None
            x = None
            y = None
            for y_idx in range(self.y):
                for x_idx in range(self.x):
                    node_vect = self.node_vectors[y_idx, x_idx]
                    dist = self.distance(data_vect, node_vect)
                    if min_dist is None or min_dist > dist:
                        min_dist = dist
                        x = x_idx
                        y = y_idx            
            
            locations[idx, 0] = y
            locations[idx, 1] = x
            distances[idx] = min_dist

        print('Done')
        return locations, distances

    def initialize_map(self):
        # Initialize map weight vectors
        ds_mul = np.mean(self.input_arr) / 0.5
        self.node_vectors = np.random.rand(self.y, self.x, self.ch) * ds_mul

        
    def fit(self, input_arr, n_iter, batch_size=32, lr=0.25, random_sampling=1.0, 
            neighbor_dist=None, dist_method='euclidean'):
        self.input_arr = input_arr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.dist_method = dist_method
        
        start_time = time.time()
        self.initialize_map()

        # Learning rate. This defines how fast the node weights are updated. 
        self.lr = lr
        self.lr_decay = 0.8 # lr decay per iteration
        
        # Neighbor node coverage.
        # This tells that how far from the best matching node the 
        # other nodes are updated. 
        if neighbor_dist is None:
            neighbor_dist = min(self.x, self.y) / 1.3
        self.nb_dist = int(neighbor_dist) 
        
        # Rate of neighbor coverage reduction per iteration. 
        # Small values = fast decay. Large values = slow decay
        self.nb_decay = 1.5 
        
        # Pad the vector map to allow easy array processing. 
        tmp_node_vects = np.zeros((self.y + 2 * self.nb_dist, self.x + 2 * self.nb_dist, self.ch))
        tmp_node_vects[self.nb_dist : self.nb_dist + self.y, 
                       self.nb_dist : self.nb_dist + self.x] = self.node_vectors.copy()
        self.node_vectors = tmp_node_vects

        # Calculate number of data points per iteration. 
        # random_sampling can vary between 0 and 1. Together with index shuffling 
        # it can be used to randomly select fraction of data for each iteration 
        # to speed up the training. 
        if random_sampling > 1 or random_sampling <= 0:
            random_sampling = 1
        n_data_pts = int(self.input_arr.shape[0] * random_sampling)


        data_idx_arr = np.arange(self.input_arr.shape[0])
        batch_count = math.ceil(n_data_pts / self.batch_size)
        n_per_report_step = int(n_data_pts / 20)
        
        # Main iteration loop. One iteration means that all data is 
        # used once to train the map weights. 
        for iteration in range(self.n_iter):
            
            # Update the neighbor function. Typically the neighbor coverage
            # reduces with increasing iteration number
            self.make_neighbor_function(iteration)

            # Shuffle the data indexes. 
            np.random.shuffle(data_idx_arr)
            
            # Temporary variables
            total_dist = 0 
            total_count = 0
            print_count = 0
            
            # Batch processing loop. The map weight update is done at the end of 
            # each batch. Often the batch size is e.g. some tens of data samples. 
            # Too large batch size can lead to convergence problems. 
            for batch in range(batch_count): 
                
                # Calculate steps (data points) for this batch
                steps_left = n_data_pts - batch * self.batch_size
                if steps_left < self.batch_size:
                    steps_in_batch = steps_left
                else:
                    steps_in_batch = self.batch_size
                
                # Create array for storing the best matching node indexes 
                bm_node_idx_arr = np.zeros((steps_in_batch, 3), dtype=np.int32)

                # Process each input data point in this batch
                for step in range(steps_in_batch):
                    
                    # Print progress update on the screen
                    if total_count % n_per_report_step == 0: 
                        print_count += 1
                        sys.stdout.write(f'\rProcessing SOM iteration {iteration + 1}/{self.n_iter}' +\
                                 ' [' + '=' * (print_count) + '>' + '.' * (20 - print_count) + ']')
                    total_count += 1

                    # Get the input data and calculate distance to the best matching node in the map
                    input_idx = data_idx_arr[batch * self.batch_size + step]
                    input_vect = self.input_arr[input_idx]
                    y, x, dist = self.find_best_matching_node(input_vect)
                    bm_node_idx_arr[step, 0] = y
                    bm_node_idx_arr[step, 1] = x
                    bm_node_idx_arr[step, 2] = input_idx 
                    total_dist += dist
                
                # Update the map weights at the end of the batch
                self.update_node_vectors(bm_node_idx_arr)
                
            # Print the average input data distance to the best matching node in the map.  
            print(f' Average distance = {total_dist / n_data_pts:0.5f}')
            
            # Update the learnig rate 
            self.lr *= self.lr_decay
            
        # Remove padding from the vector map
        self.node_vectors = self.node_vectors[self.nb_dist : self.nb_dist + self.y, 
                                              self.nb_dist : self.nb_dist + self.x]
        
        # Delete the input data array from the map instance
        del self.input_arr

        end_time = time.time()
        self.trained = True
        print(f'Training done in {end_time - start_time:0.6f} seconds.')
            
    def update_node_vectors(self, bm_node_idx_arr):
        # This method updates the map node weights. 
        # Input is one batch of best matching node coordinates and indexes to corresponsing 
        # data array locations. 
        
        for idx in range(bm_node_idx_arr.shape[0]):
            node_y = bm_node_idx_arr[idx, 0]
            node_x = bm_node_idx_arr[idx, 1]
            inp_idx = bm_node_idx_arr[idx, 2]
            input_vect = self.input_arr[inp_idx]
            
            old_coeffs = self.node_vectors[node_y + self.y_delta + self.nb_dist, node_x + self.x_delta + self.nb_dist]
            
            update_vect = self.nb_weights * self.lr * (np.expand_dims(input_vect, axis=0) - old_coeffs)
            
            self.node_vectors[node_y + self.y_delta + self.nb_dist, 
                              node_x + self.x_delta + self.nb_dist, :] += update_vect
              
    def find_best_matching_node(self, data_vect):
        # This method is used to find best matching node for data vector. 
        # The node coordinates and distance are returned. 
        # This can be used only by the fit process since this assumes that the map is 
        # padded. 
        
        min_dist = None
        x = None
        y = None
        for y_idx in range(self.y):
            for x_idx in range(self.x):
                node_vect = self.node_vectors[y_idx + self.nb_dist, x_idx + self.nb_dist]
                dist = self.distance(data_vect, node_vect)
                if min_dist is None or min_dist > dist:
                    min_dist = dist
                    x = x_idx
                    y = y_idx
                
        return y, x, min_dist
                
         
    def make_neighbor_function(self, iteration):
        # This method creates Gaussian 'bell' shaped 3D weight array in that is 
        # stored in 2D arrays. There is own array for x coordinates, y coordinates and 
        # for the weight values. The coordinate zero point is at the center of the Gaussian curve.
        # The bell width reduces when iteration value increases. 
        
        size = self.nb_dist * 2
        sigma = size / (7 + iteration / self.nb_decay)
        self.nb_weights = np.full((size * size, self.ch), 0.0)
        cp = size / 2.0 
        p1 = 1.0 / (2 * math.pi * sigma ** 2) 
        pdiv = 2.0 * sigma ** 2
        y_delta = []
        x_delta = []
        for y in range(size):
            for x in range(size):
                ep = -1.0 * ((x - cp) ** 2.0 + (y - cp) ** 2.0) / pdiv
                value = p1 * math.e ** ep
                self.nb_weights[y * size + x] = value
                y_delta.append(y - int(cp))
                x_delta.append(x - int(cp))
        self.x_delta = np.array(x_delta, dtype=np.int32)
        self.y_delta = np.array(y_delta, dtype=np.int32)

        self.nb_weights -= self.nb_weights[size // 2]
        self.nb_weights[self.nb_weights < 0] = 0 
        self.nb_weights /= np.max(self.nb_weights)  

        
    def get_umatrix(self):
        # This method creates a map of average vector distances from each node to the nodes 
        # above, below, left and right. 

        if not self.trained:
            return False

        umatrix = np.zeros((self.y, self.x))
        
        for map_y in range(self.y):
            for map_x in range(self.x):                

                n_dist = 0 
                total_dist = 0
                
                if map_y > 0: 
                    dist_up = self.distance(self.node_vectors[map_y, map_x], 
                                            self.node_vectors[map_y - 1, map_x])
                    total_dist += dist_up 
                    n_dist += 1

                if map_y < self.y - 1: 
                    dist_down = self.distance(self.node_vectors[map_y, map_x], 
                                              self.node_vectors[map_y + 1, map_x])
                    total_dist += dist_down 
                    n_dist += 1

                if map_x > 0: 
                    dist_left = self.distance(self.node_vectors[map_y, map_x], 
                                              self.node_vectors[map_y, map_x - 1])
                    total_dist += dist_left
                    n_dist += 1
                    
                if map_x < self.x - 1: 
                    dist_right = self.distance(self.node_vectors[map_y, map_x], 
                                               self.node_vectors[map_y, map_x + 1])
                    total_dist += dist_right
                    n_dist += 1

                avg_dist = total_dist / n_dist
                umatrix[map_y, map_x] = avg_dist
                
        return umatrix 
    

    def get_component_plane(self, component):
        if not self.trained:
            return False
        cplane = self.node_vectors[:, :, component].copy()
        return cplane


def plot_data_on_map(umatrix, data_locations, data_colors, data_labels=None,
                        node_width=20,
                        node_edge_color=0,
                        data_marker_size=100,
                        invert_umatrix=True,
                        plot_labels=False,
                        dpi=100):
    
    map_x = umatrix.shape[1]
    map_y = umatrix.shape[0]
    canvas = np.zeros((map_y * node_width, map_x * node_width))
    
    tmp_umatrix = umatrix.copy()
    tmp_umatrix -= np.min(tmp_umatrix)
    tmp_umatrix /= np.max(tmp_umatrix)
    
    if invert_umatrix:
        tmp_umatrix = 1 - tmp_umatrix

    for y in range(map_y):
        for x in range(map_x):
            canvas[y * node_width : (y + 1) * node_width, 
                   x * node_width : (x + 1) * node_width] = tmp_umatrix[y, x]
    
    if not node_edge_color is None:
        # Draw node borders
        for y in range(map_y):
            canvas[y * node_width, :] = node_edge_color  

        for x in range(map_x):
            canvas[:, x * node_width] = node_edge_color  
   
    # Plot the SOM u-matrix as background 
    plt.figure(figsize=(map_x * node_width / dpi, map_y * node_width / dpi), dpi=dpi)
    plt.imshow(canvas, cmap='gray', interpolation='hanning')

    # Initialize some temp variables
    item_count_map = np.zeros(umatrix.shape) 
    n_data_pts = data_locations.shape[0]

    for i in range(n_data_pts):

        x = data_locations[i, 1]
        y = data_locations[i, 0]
        items_in_cell = item_count_map[y, x]
        item_count_map[y, x] += 1
        x = x * node_width + node_width // 2 + items_in_cell * 5
        y = y * node_width + node_width // 2 + items_in_cell * 5
        plt.scatter(x, y, s=data_marker_size, color=data_colors[i], edgecolors=[0,0,0])
        
        if plot_labels:
            plt.annotate(str(data_labels[i]), (x + 8, y), size='small')

        plt.axis('off')
        
    filename = 'SOM_mapping_' + str(int(time.time())) + '.png'
    plt.savefig(filename)
    plt.show()
    print(f'Image saved to {filename}')
 