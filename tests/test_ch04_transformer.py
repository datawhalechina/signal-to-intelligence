"""Tests for Chapter 4: Transformer"""
import numpy as np


class TestSelfAttention:
    """Test self-attention mechanism"""

    def test_attention_output_shape(self, seed, load_code_module):
        """Test attention output shape"""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")
        seq_len = 10
        d_model = 64

        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        output, attention_weights = self_attention.scaled_dot_product_attention(Q, K, V)

        assert output.shape == (seq_len, d_model)
        assert attention_weights.shape == (seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, seed, load_code_module):
        """Test that attention weights sum to 1"""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")
        seq_len = 10
        d_model = 64

        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        _, attention_weights = self_attention.scaled_dot_product_attention(Q, K, V)

        # Each row should sum to 1
        assert np.allclose(np.sum(attention_weights, axis=1), 1.0)

    def test_attention_core_rejects_incompatible_shapes(self, load_code_module):
        core = load_code_module("code/ch04_transformer/attention_core.py")

        with np.testing.assert_raises(ValueError):
            core.scaled_dot_product_attention(
                np.ones((2, 3)),
                np.ones((4, 2)),
                np.ones((4, 3)),
            )

    def test_causal_mask_blocks_future_positions(self, load_code_module):
        core = load_code_module("code/ch04_transformer/attention_core.py")
        mask = core.causal_mask(4)

        assert mask.shape == (4, 4)
        future_positions = np.triu_indices(4, k=1)
        assert np.all(mask[future_positions] < 0)
        assert np.allclose(np.tril(mask), 0.0)

    def test_multihead_attention(self, seed, load_code_module):
        """Test multi-head attention"""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")
        seq_len = 10
        d_model = 64
        num_heads = 8
        d_k = d_model // num_heads

        X = np.random.randn(1, seq_len, d_model)
        W_q, W_k, W_v, W_o = self_attention.build_demo_projection_matrices(d_model, num_heads)

        output, attention_weights = self_attention.multi_head_attention(
            X, W_q, W_k, W_v, W_o, num_heads=num_heads
        )

        assert d_k == 8
        assert output.shape == (seq_len, d_model)
        assert len(attention_weights) == num_heads
        assert all(weights.shape == (seq_len, seq_len) for weights in attention_weights)

    def test_run_experiment_returns_consistent_shapes(self, load_code_module):
        """Test the reusable experiment entry point."""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")

        result = self_attention.run_experiment(seq_len=8, d_model=32, num_heads=4, seed=42)

        assert result["X"].shape == (1, 8, 32)
        assert result["output"].shape == (8, 32)
        assert result["average_attention_weights"].shape == (8, 8)
        assert len(result["attention_weights"]) == 4


class TestTransformerBlock:
    """Test Transformer block"""

    def test_layer_norm_stability(self, seed, load_code_module):
        """Test layer normalization stability"""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")
        batch_size = 32
        seq_len = 10
        d_model = 64

        x = np.random.randn(batch_size, seq_len, d_model)
        x_norm = self_attention.layer_norm(x)

        # Check that normalized values have mean ~0 and std ~1
        assert np.allclose(np.mean(x_norm, axis=-1), 0, atol=1e-5)
        assert np.allclose(np.std(x_norm, axis=-1), 1, atol=1e-5)

    def test_feed_forward_network(self, seed, load_code_module):
        """Test feed-forward network in Transformer"""
        self_attention = load_code_module("code/ch04_transformer/self_attention.py")
        seq_len = 10
        d_model = 64
        d_ff = 256

        x = np.random.randn(seq_len, d_model)

        # Feed-forward: Linear -> ReLU -> Linear
        W1 = np.random.randn(d_model, d_ff)
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d_ff, d_model)
        b2 = np.zeros(d_model)

        output = self_attention.feed_forward_network(x, W1, b1, W2, b2)

        assert output.shape == (seq_len, d_model)


class TestTransformerDemos:
    """Test chapter 4 demo scripts"""

    def test_scaled_attention_statistics(self, load_code_module):
        scaled_attention = load_code_module(
            "code/ch04_transformer/scaled_attention_demo.py"
        )
        result = scaled_attention.run_experiment(seed=42)

        assert result["dims"].shape == (4,)
        assert result["raw_score_stds"].shape == (4,)
        assert result["scaled_score_stds"].shape == (4,)
        assert result["raw_softmax_peaks"].shape == (4,)
        assert result["scaled_softmax_peaks"].shape == (4,)
        assert np.all(result["raw_score_stds"] > result["scaled_score_stds"])
        assert np.isfinite(result["raw_score_stds"]).all()
        assert np.isfinite(result["scaled_score_stds"]).all()
        assert np.isfinite(result["raw_softmax_peaks"]).all()
        assert np.isfinite(result["scaled_softmax_peaks"]).all()

    def test_causal_mask_properties(self, load_code_module):
        causal_mask = load_code_module("code/ch04_transformer/causal_mask_demo.py")
        result = causal_mask.run_experiment(seed=42)

        assert result["scores"].shape == (8, 8)
        assert result["bidirectional_weights"].shape == (8, 8)
        assert result["causal_weights"].shape == (8, 8)
        assert result["bidirectional_output"].shape == (8, 16)
        assert result["causal_output"].shape == (8, 16)
        assert np.allclose(np.sum(result["bidirectional_weights"], axis=1), 1.0)
        assert np.allclose(np.sum(result["causal_weights"], axis=1), 1.0)
        assert np.allclose(
            np.triu(result["causal_weights"], k=1),
            0.0,
            atol=1e-8,
        )
        assert np.isfinite(result["output_diffs"]).all()
        assert np.any(result["output_diffs"] > 0)

    def test_graph_theory_demo_properties(self, load_code_module):
        graph_demo = load_code_module("code/ch04_transformer/graph_theory_demo.py")
        result = graph_demo.run_experiment(seed=42)

        assert result["adjacency_matrix"].shape == (5, 5)
        assert np.allclose(result["adjacency_matrix"], result["adjacency_matrix"].T)
        assert result["dfs_order"][0] == 0
        assert result["bfs_order"][0] == 0
        assert len(result["dfs_order"]) == 5
        assert len(result["bfs_order"]) == 5
        assert result["attention_weights"].shape == (4, 4)
        assert np.isclose(result["eigenvalues"][0], 0.0)
        assert result["is_connected"] is True
        assert result["connected_components"] == 1
        assert (0, 4) in result["shortest_paths"]
