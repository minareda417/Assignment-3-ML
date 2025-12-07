import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node:
    def __init__(self, 
                 feature_idx =None, 
                 threshold=None, 
                 left=None, 
                 right=None, 
                 value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left # <=
        self.right = right # >
        self.value = value # for leaf nodes

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, max_features: int = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def __calc_entropy(self, y:np.ndarray):
        hist = np.bincount(y) 
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])

    def __calc_information_gain(self, X:np.ndarray, y:np.ndarray, feature_idx, threshold):
        # calculate entropy before split
        parent_entropy = self.__calc_entropy(y)

        # calculate entropy after split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_partition = y[left_mask]
        right_partition = y[right_mask]

        if len(left_partition) == 0 or len(right_partition) == 0:
            return 0

        left_entropy = self.__calc_entropy(left_partition)
        right_entropy = self.__calc_entropy(right_partition)
        
        left_p = len(left_partition)/len(y)
        right_p = len(right_partition)/len(y)
        
        child_entropy = (left_p * left_entropy + right_p * right_entropy)
        ig = parent_entropy - child_entropy
        return ig

    def __get_possible_thresholds(self, X:np.ndarray, feature_idx):
        x_col = X[:, feature_idx]
        x_col = np.unique(x_col)
        x_col = np.sort(x_col)

        if(x_col.shape[0] <= 1):
            return [x_col[0]]
        return (x_col[1:] + x_col[:-1])/2.0

    def __find_best_split(self, X:np.ndarray, y:np.ndarray):
        max_ig = -1
        max_ig_feature, max_ig_threshold = None, None
        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features,
                                               size=min(self.max_features, n_features),
                                               replace=False)
        else:
            feature_indices = range(n_features)
        for feature_idx in feature_indices:
            ts = self.__get_possible_thresholds(X, feature_idx)
            for t in ts:
                ig = self.__calc_information_gain(X, y, feature_idx, t)
                if ig > max_ig:
                    max_ig = ig
                    max_ig_feature = feature_idx
                    max_ig_threshold = t

        return max_ig_feature, max_ig_threshold 

    def __grow_tree(self, X: np.ndarray, y:np.ndarray, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        
        if(num_samples < self.min_samples_split or 
           depth == self.max_depth or
           num_labels == 1):
            value = np.argmax(np.bincount(y))
            return Node(value=value)


        feature_idx, threshold = self.__find_best_split(X, y)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if(left_mask.sum() == 0 or right_mask.sum() == 0):
            value = np.argmax(np.bincount(y))
            return Node(value=value)

        x_left = X[left_mask]
        y_left = y[left_mask]

        x_right = X[right_mask]
        y_right = y[right_mask]
        
        left_node = self.__grow_tree(x_left, y_left, depth+1)
        right_node = self.__grow_tree(x_right, y_right, depth+1)

        return Node(feature_idx, threshold, left_node, right_node)

    def fit(self, X, y):
        self.root = self.__grow_tree(X, y)

    def predict(self, X):
       return np.array([self.__traverse_tree(x) for x in X])

    def __traverse_tree(self, x):
        if self.root is None:
            raise Exception("The model is not trained")
        curr: Node = self.root
        while not curr.is_leaf():
            feat_idx = curr.feature_idx
            threshold = curr.threshold

            if x[feat_idx] <= threshold:
                curr = curr.left
            else:
                curr = curr.right

        return curr.value

    def plot_tree(self, feature_names=None, class_names=None, max_depth=None):
        """
        Plot the decision tree structure.
        
        Parameters:
        - feature_names: List of feature names (optional)
        - class_names: List of class names (optional)
        - max_depth: Maximum depth to visualize (optional, plots full tree if None)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if self.root is None:
            raise Exception("The model is not trained")
        
        def get_tree_depth(node):
            if node is None or node.is_leaf():
                return 0
            return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))
        
        def count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return count_leaves(node.left) + count_leaves(node.right)
        
        total_depth = get_tree_depth(self.root)
        if max_depth is not None:
            total_depth = min(total_depth, max_depth)
        
        num_leaves = count_leaves(self.root)
        
        # Better figure sizing
        fig_width = max(15, num_leaves * 2)
        fig_height = max(10, total_depth * 2.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        
        # Assign x positions to leaves using in-order traversal
        self._leaf_counter = 0
        positions = {}
        
        def get_positions(node, depth=0):
            if node is None or (max_depth is not None and depth > max_depth):
                return 0
            
            if node.is_leaf() or (max_depth is not None and depth == max_depth):
                self._leaf_counter += 1
                x_pos = self._leaf_counter
                positions[id(node)] = (x_pos, total_depth - depth)
                return x_pos
            
            left_x = get_positions(node.left, depth + 1)
            right_x = get_positions(node.right, depth + 1)
            x_pos = (left_x + right_x) / 2
            positions[id(node)] = (x_pos, total_depth - depth)
            
            return x_pos
        
        get_positions(self.root)
        
        # Set axis limits
        ax.set_xlim(0, self._leaf_counter + 1)
        ax.set_ylim(-0.5, total_depth + 0.5)
        
        def plot_node(node, depth=0):
            if node is None or (max_depth is not None and depth > max_depth):
                return
            
            x, y = positions[id(node)]
            
            box_width = 0.8
            box_height = 0.6
            
            if node.is_leaf() or (max_depth is not None and depth == max_depth):
                # Leaf node
                class_label = class_names[node.value] if class_names is not None else str(node.value)
                box = patches.FancyBboxPatch((x - box_width/2, y - box_height/2), 
                                            box_width, box_height,
                                            boxstyle="round,pad=0.05", 
                                            edgecolor='green', facecolor='lightgreen',
                                            linewidth=2)
                ax.add_patch(box)
                ax.text(x, y, class_label, 
                       ha='center', va='center', fontsize=10, weight='bold')
            else:
                # Decision node
                feature_name = feature_names[node.feature_idx] if feature_names is not None else f"X[{node.feature_idx}]"
                if len(feature_name) > 20:
                    feature_name = feature_name[:17] + "..."
                label = f"{feature_name}\n≤ {node.threshold:.2f}"
                
                box = patches.FancyBboxPatch((x - box_width/2, y - box_height/2), 
                                            box_width, box_height,
                                            boxstyle="round,pad=0.05", 
                                            edgecolor='blue', facecolor='lightblue',
                                            linewidth=2)
                ax.add_patch(box)
                ax.text(x, y, label, 
                       ha='center', va='center', fontsize=9)
                
                # Draw edges to children
                if node.left and depth < total_depth:
                    x_left, y_left = positions[id(node.left)]
                    ax.plot([x, x_left], [y - box_height/2, y_left + box_height/2], 
                           'k-', linewidth=2, alpha=0.7)
                    ax.text((x + x_left)/2 - 0.1, (y + y_left)/2, 'Yes', 
                           fontsize=8, color='green', weight='bold')
                    plot_node(node.left, depth + 1)
                
                if node.right and depth < total_depth:
                    x_right, y_right = positions[id(node.right)]
                    ax.plot([x, x_right], [y - box_height/2, y_right + box_height/2], 
                           'k-', linewidth=2, alpha=0.7)
                    ax.text((x + x_right)/2 + 0.1, (y + y_right)/2, 'No', 
                           fontsize=8, color='red', weight='bold')
                    plot_node(node.right, depth + 1)
        
        plot_node(self.root)
        
        title = "Decision Tree Visualization"
        if max_depth is not None:
            title += f" (showing depth ≤ {max_depth})"
        plt.title(title, fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTree(max_depth=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # check accuracy

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")
    model.plot_tree(
    feature_names=data.feature_names,
    class_names=['Malignant', 'Benign']
    )