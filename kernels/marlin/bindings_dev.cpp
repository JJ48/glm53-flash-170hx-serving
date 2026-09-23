// Standalone registration of the Marlin MoE GEMM under its own namespace (_marlin_dev), same schema as the fork's
// _moe_C::moe_wna16_marlin_gemm (upstream 8e92248f79, before the g_idx/perm removal), so it can be swapped in from
// Python behind an env flag and A/B'd against the shipped kernel. Built with torch's stable ABI headers.
#include <optional>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "core/scalar_type.hpp"

// Declaration matching the definition in marlin_moe_wna16/ops.cu (upstream 8e92248f79).
torch::stable::Tensor moe_wna16_marlin_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    std::optional<torch::stable::Tensor> const& g_idx_or_none,
    std::optional<torch::stable::Tensor> const& perm_or_none,
    torch::stable::Tensor& workspace, torch::stable::Tensor& sorted_token_ids,
    torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& num_tokens_past_padded,
    torch::stable::Tensor& topk_weights, int64_t moe_block_size, int64_t top_k,
    bool mul_topk_weights, vllm::ScalarTypeId const& b_type_id, int64_t size_m,
    int64_t size_n, int64_t size_k, bool is_k_full, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float, int64_t thread_k, int64_t thread_n,
    int64_t blocks_per_sm);

STABLE_TORCH_LIBRARY(_marlin_dev, m) {
  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k, bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_marlin_dev, CUDA, m) {
  m.impl("moe_wna16_marlin_gemm", TORCH_BOX(&moe_wna16_marlin_gemm));
}
