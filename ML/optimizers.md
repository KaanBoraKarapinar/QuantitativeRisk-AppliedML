
### SGD with Momentum

Accumulates a velocity vector to dampen oscillations and accelerate consistent directions:

$$v \leftarrow \beta v + \frac{\partial \mathcal{L}}{\partial w}, \quad w \leftarrow w - \eta \cdot v$$

### Adam

Combines momentum with adaptive per-parameter learning rates.
Maintains a running mean and variance of gradients:

$$w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

Default choice for most tasks. Works well out of the box with $\eta = 0.001$.

### AdamW

Adam with decoupled weight decay. Standard for Transformers and LLMs.