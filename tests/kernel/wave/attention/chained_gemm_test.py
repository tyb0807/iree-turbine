# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_ones,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    require_cdna3,
    param_bool,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes


@require_e2e
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 128, 128, 128)]
 #  get_test_shapes("chained_gemm") +
 #  [(1, m, k, k, n)
 #   for m in [1, 33, 64, 222]
 #   for n in [128, 1024, 2048]
 #   for k in [128, 512, 1024]]
   #[(1, 128, 128, 128, 128)]
   #[(1, m, n, k1, k2)
   # for m in [1, 33, 64, 222]
   # for n in [128, 256, 512, 1024]
   # for k1 in [128, 256, 320] # 511, 512, 1024
   # for k2 in [128, 1024, 2048]]
 #  _, m, n, k1, k2 = shape
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    #   MMAType.F32_32x32x8_F16,
    ],
)
def testMoe(
    shape: tuple[int],
    dtype,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
    #       vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def moe(
        a: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        w1_gate: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        w1_up: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        w2: tkl.Memory[N, K2, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        out1: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        out2: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        out3: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        out4: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        out1_reg = tkl.Register[M, N, tkl.f32](0.0)
        out2_reg = tkl.Register[M, N, tkl.f32](0.0)
        out3_reg = tkl.Register[M, N, tkl.f32](0.0)
        out4_reg = tkl.Register[M, N, tkl.f32](0.0)

        cst_m1 = tkl.Register[M, K2, tkl.f32](-1.0)
        cst_1 = tkl.Register[M, K2, tkl.f32](1.0)
        cst_2 = tkl.Register[M, K2, tkl.f32](2.0)

        @tkw.iterate(K2, init_args=[out1_reg, out2_reg, out3_reg, out4_reg])
        def repeat(
            acc1: tkl.Register[M, N, tkl.f32],
            acc2: tkl.Register[M, N, tkl.f32],
            acc3: tkl.Register[M, N, tkl.f32],
            acc4: tkl.Register[M, N, tkl.f32],
        ) -> (
            tkl.Register[M, N, tkl.f32],
            tkl.Register[M, N, tkl.f32],
            tkl.Register[M, N, tkl.f32],
            tkl.Register[M, N, tkl.f32],
        ):
            gate_acc = tkl.Register[K2, M, tkl.f32](0.0)
            up_acc = tkl.Register[K2, M, tkl.f32](0.0)

            a_reg = tkw.read(a)
            w1_gate_reg = tkw.read(w1_gate)
            gate_reg = tkw.mma(w1_gate_reg, a_reg, gate_acc)
            gate_reg_permuted = tkw.permute(gate_reg, target_shape=[M, K2])

            w1_up_reg = tkw.read(w1_up)
            up_reg = tkw.mma(w1_up_reg, a_reg, up_acc)
            up_reg_permuted = tkw.permute(up_reg, target_shape=[M, K2])

            minus_gate = gate_reg_permuted * cst_m1
          # exp_out = tkw.exp(minus_gate)
          # exp_out = gate_reg / cst_2 + up_reg / cst_2
            exp_out = gate_reg_permuted + cst_1
          # exp_out = gate_reg_permuted + up_reg_permuted
          # exp_out = tkw.permute(exp_out, target_shape=[M, K2])
          # sigmoid = gate_reg / (cst_1 + exp_out)

          # res_reg = sigmoid * up_reg
            gate_cast_reg = tkw.cast(gate_reg_permuted, tkl.bf16)
            up_cast_reg = tkw.cast(up_reg_permuted, tkl.bf16)
            minus_gate = tkw.cast(minus_gate, tkl.bf16)
            exp_out = tkw.cast(exp_out, tkl.bf16)
          # sigmoid = tkw.cast(sigmoid, tkl.bf16)
          # res_cast_reg = tkw.cast(res_reg, tkl.bf16)

            w2_reg = tkw.read(w2)
            acc1 = tkw.mma(gate_cast_reg, w2_reg, acc1)
            acc2 = tkw.mma(up_cast_reg, w2_reg, acc2)
            acc3 = tkw.mma(minus_gate, w2_reg, acc3)
            acc4 = tkw.mma(exp_out, w2_reg, acc4)
          # acc3 = tkw.mma(res_cast_reg, w2_reg, acc3)

            return acc1, acc2, acc3, acc4

        # repeat represents the results of the loop
        res1, res2, res3, res4 = repeat
        tkw.write(res1, out1)
        tkw.write(res2, out2)
        tkw.write(res3, out3)
        tkw.write(res4, out4)

    _, m, n, k1, k2 = shape
    print(shape)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        M: m,
        N: n,
        K1: k1,
        K2: k2,
    }
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
     #  denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
     #  print_ir_after_all=True,
    )
    options = set_default_run_config(options)
    moe = wave_compile(options, moe)

    a = device_randn(m, k1, dtype=torch.bfloat16)
    w1_gate = device_ones(k2, k1, dtype=torch.bfloat16)
    w1_up = device_randn(k2, k1, dtype=torch.bfloat16)
    w2 = device_randn(n, k2, dtype=torch.bfloat16)
    output1 = device_zeros(m, n, dtype=torch.float32)
    output2 = device_zeros(m, n, dtype=torch.float32)
    output3 = device_zeros(m, n, dtype=torch.float32)
    output4 = device_zeros(m, n, dtype=torch.float32)
    asm = moe(a, w1_gate, w1_up, w2, output1, output2, output3, output4)

    filename = f"wave_moe_{'x'.join(map(str, shape))}.mlir"
    with open(filename, "w") as f:
        f.write(asm)

    a = a.to(torch.float32)
    w1_gate = w1_gate.to(torch.float32)
    w1_up = w1_up.to(torch.float32)
    w2 = w2.to(torch.float32)
    gate = torch.matmul(a, w1_gate.transpose(-1, -2))
    up = torch.matmul(a, w1_up.transpose(-1, -2))
    torch_ref1 = torch.matmul(gate, w2.transpose(-1, -2))
    torch_ref2 = torch.matmul(up, w2.transpose(-1, -2))
    silu = gate / (1 + torch.exp(-gate))
    silu_and_mul = silu * up
    torch_ref3 = torch.matmul(-gate, w2.transpose(-1, -2))
  # torch_ref4 = torch.matmul(torch.exp(-gate), w2.transpose(-1, -2))
  # torch_ref4 = torch.matmul(gate / 2 + up / 2, w2.transpose(-1, -2))
    torch_ref4 = torch.matmul(gate + 1, w2.transpose(-1, -2))
  # torch_ref3 = torch.matmul(silu_and_mul, w2.transpose(-1, -2))
  # output1 = output1.to(torch.bfloat16)
  # output2 = output2.to(torch.bfloat16)
  # output3 = output3.to(torch.bfloat16)
  # output4 = output4.to(torch.bfloat16)
    # only 1-4 elements mismatched, small margin (e.g. 77 vs. 77.5)
    for i in range(m):
        for j in range(n):
            try:
                assert_close(output2[i][j], torch_ref2[i][j], atol=1e-1, rtol=1e-2)
            except AssertionError:
                print(f"res[{i}][{j}] = {output4[i][j]} vs. {torch_ref4[i][j]}")
    assert_close(output1, torch_ref1, atol=5e-2, rtol=5e-3)
    assert_close(output2, torch_ref2, atol=5e-2, rtol=5e-3)
    assert_close(output3, torch_ref3, atol=5e-2, rtol=5e-3)
    assert_close(output4, torch_ref4, atol=1e-1, rtol=1e-2)


@require_e2e
@pytest.mark.parametrize(
    "shape",
 #  [(1, 1, 128, 128, 128)] +
 #  get_test_shapes("chained_gemm") +
 #  [(1, m, k, k, n)
 #   for m in [1, 33, 64, 222]
 #   for n in [128, 1024, 2048]
 #   for k in [128, 512, 1024]]
    [(1, m, n, k1, k2)
     for m in [1, 33, 64, 222]
     for n in [128, 256, 512, 1024]
     for k1 in [128, 256, 320] # 511, 512, 1024
     for k2 in [128, 1024, 2048]]
 #  _, m, n, k1, k2 = shape
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    #   MMAType.F32_32x32x8_F16,
    ],
)
def testElementwise(
    shape: tuple[int],
    dtype,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    _, m, n, k1, k2 = shape
    print(shape)

    # Expose user-constraints
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(n, 256), wave_size)

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def elementwise(
        x1: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.bf16],
      # x2: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, datatype],
        out: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.bf16],
    ):
        x1_reg = tkw.read(x1)
        cst_m1 = tkl.Register[M, N, tkl.bf16](-1.0)
      # cst_1 = tkl.Register[M, N, tkl.bf16](1.0)
      # exp_out = cst_1 / x1_reg
        exp_out = tkw.exp(cst_m1 * x1_reg)
      # sigmoid = cst_1 / (cst_1 + exp_out)
      # silu = sigmoid * x1_reg

      # x2_reg = tkw.read(x2)
      # res = silu * x2_reg

        tkw.write(exp_out, out)

    hyperparams = {
        M: m,
        N: n,
    }
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        wave_runtime=False,
        use_scheduling_barriers=enable_scheduling_barriers,
     #  denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
     #  print_ir_after_all=True,
    )
    options = set_default_run_config(options)
    elementwise = wave_compile(options, elementwise)

    gate = device_randn(m, n, dtype=torch.bfloat16)
    output4 = device_zeros(m, n, dtype=torch.bfloat16)
    asm = elementwise(gate, output4)

    filename = f"wave_elementwise_{'x'.join(map(str, shape))}.mlir"
    with open(filename, "w") as f:
        f.write(asm)

    torch_ref4 = 1 / gate
    torch_ref4 = torch.exp(-gate)
    for i in range(m):
        for j in range(n):
            try:
                assert_close(output4[i][j], torch_ref4[i][j], atol=1e-1, rtol=1e-2)
            except AssertionError:
                print(f"res[{i}][{j}] = {output4[i][j]} vs. {torch_ref4[i][j]}")
    assert_close(output4, torch_ref4, atol=5e-2, rtol=5e-3)


@require_e2e
#@pytest.mark.parametrize("shape",
#    [
#    #   (1, 33, 63, 64, 128),
#    #   (1, 33, 63, 65, 128),
#    #   (1, 33, 63, 126, 128),
#    #   (1, 33, 63, 127, 128),
#    #   (1, 33, 63, 128, 128),
#    #   (1, 33, 63, 129, 128),
#    #   (1, 33, 63, 159, 128),
#    #   (1, 33, 63, 160, 128),
#    #   (1, 33, 63, 176, 128),
#        (1, 222, 1024, 1024, 2048),
#    ]
#)
@pytest.mark.parametrize(
    "shape",
 #  [(1, 1, 128, 128, 128)] +
 #  get_test_shapes("chained_gemm") +
    [(1, m, n, k1, k2)
     for m in [1, 33, 64, 222]
     for n in [128, 256, 512, 1024]
     for k1 in [128, 256, 320] # 511, 512, 1024
     for k2 in [128, 1024, 2048]]
 #  _, m, n, k1, k2 = shape
)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
 #      MMAType.F32_32x32x8_F16,
    ],
)
def testChainedGemm(
    shape: tuple[int],
    dtype,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        k: tkl.Memory[B, K2, K1, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        v: tkl.Memory[B, N, K2, GLOBAL_ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.bf16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_scheduling_barriers=enable_scheduling_barriers,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
    )
    options = set_default_run_config(options)
    chained_gemm = wave_compile(options, chained_gemm)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.bfloat16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.bfloat16)
    v = device_randn(batch, v_head_dim, kv_seq_len, dtype=torch.bfloat16)
    output = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    asm = chained_gemm(q, k, v, output)

    dump_generated_mlir = True
    if dump_generated_mlir:
        filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)
            print(f"IR dumped to {filename}")

  # iree_ref = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
  # generate_iree_ref("chain_mmt", [q, k, v], [iree_ref])
  # assert_close(output, iree_ref, check_device=False, atol=0, rtol=0)

    torch_qk = torch.matmul(q, k.transpose(-1, -2))
    torch_ref = torch.matmul(torch_qk, v.transpose(-1, -2))
    output_for_cmp = output.transpose(-1, -2).to(torch.bfloat16)

    print("SHAPE ", torch_ref.shape)
    for i in range(len(torch_ref[0])):
        for j in range(len(torch_ref[0][0])):
            try:
                assert_close(output_for_cmp[0][i][j], torch_ref[0][i][j], atol=8e-2, rtol=8e-3)
            except AssertionError:
                print(f"{i} {j} {output_for_cmp[0][i][j]} vs. {torch_ref[0][i][j]}")
    assert_close(output_for_cmp, torch_ref, atol=8e-2, rtol=8e-3)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("chained_gemm"))
@param_bool("enable_scheduling", "sched", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        pytest.param(
            (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
            id="MFMA_32x32x16+32x32x16_K4",
        ),
        pytest.param(
            (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
            id="MFMA_16x16x32+16x16x32_K4",
        ),
    ],
)
def testChainedGemmF8(
    shape: tuple[int],
    enable_scheduling: bool,
    mfma_variant: tuple[MMAType, MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm_f8(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(qk_cast_reg, v_reg, acc, mfma_variant[1])
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_scheduling_barriers=enable_scheduling_barriers,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    chained_gemm_f8 = wave_compile(options, chained_gemm_f8)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, v_head_dim, kv_seq_len, dtype=torch.float16)
    output = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    asm = chained_gemm_f8(q, k, v, output)

    if dump_generated_mlir:
        filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    iree_ref = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
    generate_iree_ref("chain_mmt_f8", [q, k, v], [iree_ref])
    assert_close(output, iree_ref, atol=7e-5, rtol=2e-3, check_device=False)
