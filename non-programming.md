### 1. **Hadamard Matrix Product**

The Hadamard product, also known as the element-wise product, is the multiplication of two matrices of the same dimensions, where each element in the resulting matrix is the product of the corresponding elements of the input matrices. If matrices \( A \) and \( B \) have dimensions \( m \times n \), the Hadamard product \( C = A \circ B \) is calculated as:
\[
C_{ij} = A_{ij} \times B_{ij}
\]
The result is also an \( m \times n \) matrix. This operation is different from traditional matrix multiplication, as it only involves element-wise operations rather than summing products of rows and columns.

### 2. **Matrix Multiplication**

Matrix multiplication is a binary operation that takes two matrices and produces another matrix. If matrix \( A \) has dimensions \( m \times n \) and matrix \( B \) has dimensions \( n \times p \), their matrix product \( C = A \times B \) is calculated as:
\[
C_{ij} = \sum_{k=1}^{n} A_{ik} \times B_{kj}
\]
Each element of the resulting matrix \( C \) is the dot product of a row from matrix \( A \) and a column from matrix \( B \). For matrix multiplication to be valid, the number of columns in the first matrix must equal the number of rows in the second matrix. The resulting matrix has dimensions \( m \times p \).

### 3. **Transpose Matrix and Vector**

The transpose of a matrix \( A \), denoted \( A^T \), is obtained by swapping the rows and columns of \( A \). If matrix \( A \) has dimensions \( m \times n \), then its transpose \( A^T \) will have dimensions \( n \times m \), and:
\[
(A^T)_{ij} = A_{ji}
\]
For a vector (a matrix with a single row or column), the transpose changes a column vector into a row vector and vice versa.

### 4. **Training Set Batch**

In machine learning, a **batch** refers to a subset of the training data used to compute the model's gradients during one iteration of the training process. Instead of using the entire training set (which may be computationally expensive), the model is trained on smaller batches. This process is known as **mini-batch gradient descent**, and the size of the batch determines how many examples are used in a single iteration.

### 5. **Entropy-Based Loss Function**

The **entropy-based loss function** often refers to **cross-entropy loss**, which is commonly used in classification tasks. For binary classification, the cross-entropy loss is:
\[
L = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
\]
Where:
- \( y \) is the true label (0 or 1),
- \( \hat{y} \) is the predicted probability of the label being 1.

For multi-class classification, the general form of cross-entropy loss is:
\[
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
\]
Where:
- \( C \) is the number of classes,
- \( y_i \) is the true label (one-hot encoded),
- \( \hat{y}_i \) is the predicted probability for class \( i \).

This loss function is used because it measures the divergence between the true distribution (labels) and the predicted probability distribution, effectively penalizing incorrect predictions.

### 6. **Neural Network Supervised Training Process**

In **supervised learning**, the neural network is trained using labeled data. The process involves the following steps:
1. **Initialize weights**: The network's weights are initialized randomly or using some heuristic.
2. **Forward propagation**: Input data is passed through the network layer by layer to produce an output.
3. **Calculate loss**: The output is compared to the true label using a loss function (e.g., cross-entropy loss).
4. **Backpropagation**: The loss is propagated back through the network to calculate gradients with respect to each weight.
5. **Update weights**: Weights are updated using an optimization algorithm like stochastic gradient descent (SGD) to minimize the loss.
6. **Repeat**: This process is repeated over multiple iterations (epochs) until the network's performance is satisfactory.

### 7. **Forward Propagation**

**Forward propagation** is the process of moving inputs through the neural network to generate an output. In each layer:
- The input from the previous layer is multiplied by the weights of the current layer.
- A bias is added to the result.
- An activation function (e.g., ReLU, sigmoid) is applied to introduce non-linearity.
This process continues until the final layer, which produces the output (predicted values).

### 8. **Backpropagation**

**Backpropagation** is the method used to compute the gradients of the loss function with respect to the network's weights. It involves:
1. **Calculate the error**: The loss function provides the error between the predicted output and the actual label.
2. **Compute gradients**: The gradient of the loss is computed with respect to the output, which is then propagated backward through the network using the chain rule.
3. **Update weights**: Using the computed gradients, the weights are adjusted to minimize the loss. The weights are updated using an optimization algorithm like SGD.

The combination of forward propagation and backpropagation allows the neural network to learn from data and optimize its parameters to improve predictions.
