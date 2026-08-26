#include "codegen_tang.h"

#include "runtime/pack_args.h"
#include "runtime/tang/tang_module.h"
#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

using namespace ffi;

static Map<String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule &mod) {
  Map<String, runtime::FunctionInfo> fmap;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IRModule with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    Array<DLDataType> arg_types;
    Array<String> launch_param_tags;
    for (const Var &param : f->params) {
      if (param->dtype.is_handle()) {
        const auto *ptr = param->type_annotation.as<PointerTypeNode>();
        if (ptr != nullptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = param.dtype();
      arg_types.push_back(dtype.is_bool() ? DataType::Int(32) : dtype);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const String &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    String name = f->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
    fmap.Set(name,
             runtime::FunctionInfo(name, arg_types, launch_param_tags, {}));
  }
  return fmap;
}

static std::string GenerateTANGSource(const IRModule &mod,
                                      const Target &target) {
  With<Target> target_scope(target);
  CodeGenTileLangTANG cg;
  cg.Init(false);
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangTANG can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    ICHECK_EQ(f->GetAttr<Integer>(tvm::attr::kCallingConv).value(),
              CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(Downcast<GlobalVar>(kv.first), f);
  }
  std::string code = cg.Finish();
  if (auto postproc = Function::GetGlobal("tilelang_callback_tang_postproc")) {
    code = (*postproc)(code, target).cast<std::string>();
  }
  return code;
}

Module BuildTileLangTANG(IRModule mod, Target target) {
  std::string code = GenerateTANGSource(mod, target);
  auto compile = Function::GetGlobal("tilelang_callback_tang_compile");
  ICHECK(compile.has_value()) << "tilelang_callback_tang_compile is not set";
  tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
  Bytes binary = (*compile)(code, target, pass_ctx->config).cast<Bytes>();
  Map<String, String> source_map;
  source_map.Set("tang", code);
  return runtime::TANGModuleCreate(binary, String("t"), ExtractFuncInfo(mod),
                                   source_map);
}

Module BuildTileLangTANGWithoutCompile(IRModule mod, Target target) {
  std::string code = GenerateTANGSource(mod, target);
  static constexpr char kDummyBinary[] = "llir";
  Map<String, String> source_map;
  source_map.Set("tang", code);
  return runtime::TANGModuleCreate(
      Bytes(kDummyBinary, sizeof(kDummyBinary) - 1), String("llir"),
      ExtractFuncInfo(mod), source_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_tang", BuildTileLangTANG)
      .def("target.build.tang", BuildTileLangTANG)
      .def("target.build.tilelang_tang_without_compile",
           BuildTileLangTANGWithoutCompile);
}

} // namespace codegen
} // namespace tvm
