#include "hetu/graph/ops/EmbeddingLookup.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void EmbeddingLookupOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  NDArray::embedding(inputs.at(0), inputs.at(1), 
                     op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList EmbeddingLookupOpImpl::DoGradient(Operator& op,
                                             const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeEmbeddingLookupGradientOp(grad_outputs.at(0), op->input(1), op->output(0), op->input(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList
EmbeddingLookupOpImpl::DoInferShape(Operator& op,
                                    const HTShapeList& input_shapes,
                                    RuntimeContext& ctx) const {
  HTShape output_shape = input_shapes[1];
  output_shape.emplace_back(input_shapes[0][1]);
  return {output_shape};
}

void EmbeddingLookupOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                           const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_id.is_valid() && 
            ds_input.get_device_num() == ds_id.get_device_num()) 
    << "EmbeddingLookupOpDef: distributed states for input and id must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1) 
    << "Tensor input and id shouldn't be partial";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Tensor input(embedding table) cannot be splited!";
  outputs.at(0)->set_distributed_states(ds_id);
}

void EmbeddingLookupGradientOpImpl::DoCompute(Operator& op,
                                              const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::EmbeddingLookupGradient,
    inputs.at(0), inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
EmbeddingLookupGradientOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext &ctx) const {
  return {input_shapes.at(3)};
}

void EmbeddingLookupGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                   const OpMeta& op_meta) const {
  const DistributedStates& ds_grad_output = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_ori_output = inputs.at(1)->get_distributed_states();
  int32_t device_num = ds_grad_output.get_device_num();
  HT_ASSERT(ds_grad_output.is_valid() && ds_id.is_valid() && ds_ori_output.is_valid()) 
    << "EmbeddingLookupGradientOpDef: distributed states for grad_output and id and ori_output must be valid!";
  HT_ASSERT(ds_grad_output.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1 && ds_ori_output.get_dim(-2) == 1) 
    << "Tensor grad_output and id and ori_output shouldn't be partial";
  HT_ASSERT(ds_grad_output.check_equal(ds_id) && ds_id.check_equal(ds_ori_output))
    << "Distributed states for tensor grad_output and id and ori_output must be equal!";
  
  outputs.at(0)->set_distributed_states({device_num, {{-2, device_num}, {-1, 1}}, {-2}}); // pure partial
}

Tensor MakeEmbeddingLookupOp(Tensor input, Tensor id, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupOpImpl>(),
          {std::move(input), std::move(id)},
          std::move(op_meta))->output(0);
}

Tensor MakeEmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input, Tensor input,
                                     OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupGradientOpImpl>(),
          {std::move(grad_output), std::move(id), std::move(ori_input), std::move(input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu