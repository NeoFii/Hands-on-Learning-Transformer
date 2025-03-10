# **Hands-on Learning: Transformer**

[简体中文](./README.md) | English

## **🚀 Project Introduction**

This project reconstructs the Transformer architecture proposed in the paper "**Attention is All You Need**" from scratch. Through step-by-step code implementation and theoretical analysis, we deeply explore the design principles and implementation details of this revolutionary model.

## **📚 Background**

The Transformer model proposed in 2017 completely changed the paradigm of sequence modeling:

- ⚡ Entirely based on attention mechanisms, abandoning the temporal dependencies of RNN/CNN
- 🚄 Friendly to parallel computing, significantly improving training efficiency
- 🏗️ Becoming the cornerstone of milestone models such as BERT and GPT

## **🧭 Learning Path**

| **Stage** | **Topic**                                    | **Status** |
| --------- | -------------------------------------------- | ---------- |
| Day 1     | 📥 Input Representation & Positional Encoding | ✅          |
| Day 2     | 👁️ Self-Attention Mechanism                   | 🔜          |
| Day 3     | 🔍 Encoder Module                             | 🔜          |
| Day 4     | 🔮 Decoder Module                             | 🔜          |
| Day 5     | 🏆 Complete Transformer                       | 🔜          |

## **🔍 Day 1: Input Representation & Positional Encoding**

- Input Embedding Layer 

  - ✨ Learnable mapping from tokens to vectors
  - 🔢 Scaling factor `sqrt(d_model)`

- Positional Encoding 

  - 📐 Implementation of sine/cosine positional encoding formula

  ```
  PE(pos,2i) = sin(pos/10000^(2i/d_model))
  PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
  ```

  - 🧮 Mathematical representation of positional information
  - 🔄 Advantages of relative positional encoding

## **👁️ Day 2: Self-Attention Mechanism (🔜 Coming Soon)**

- Multi-Head Attention
  - 🔄 Implementation of parallel attention heads
  - 🔗 Computation flow of attention matrices
- Scaled Dot-Product Attention
  - 📏 Necessity of the scaling factor
  - 🧮 Numerical stability of Softmax

## **🔧 Day 3: Encoder Module (🔜 Coming Soon)**

- Feed-Forward Neural Network
  - 🧠 Implementation of position-wise feed-forward networks
  - 📈 Choice of activation functions
- Residual Connections & Layer Normalization
  - 🔄 Importance of residual mechanisms
  - 📊 Theoretical foundation of normalization strategies

## **🔮 Day 4: Decoder Module (🔜 Coming Soon)**

- Masked Self-Attention
  - 🔒 Implementation of future information masking
  - 👁️ Special characteristics of decoder attention
- Encoder-Decoder Attention
  - 🔄 Cross-attention mechanism
  - 🔍 Information flow of Query-Key-Value

## **🏆 Day 5: Complete Transformer (🔜 Coming Soon)**

- Model Integration
  - 🧩 Assembly of the complete architecture
  - 🔄 Forward propagation process
- Training & Optimization
  - 📉 Loss function design
  - 🔧 Parameter initialization strategies

## **📚 References**

- 📑 **Attention is All You Need**

## **🤝 Contribution Guidelines**

Contributions to this project are welcome!

- 🐛 Found issues or have suggestions? Submit an issue
- 💡 Want to improve code or documentation? Submit a PR
- 📚 Share learning experiences or implementation ideas? Join the discussion

## **📄 License**

MIT License

**If this project has been helpful to you, please give it a ⭐️ to encourage the author to continue creating!**