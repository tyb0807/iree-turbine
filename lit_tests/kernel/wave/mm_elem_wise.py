# RUN: python %s | FileCheck %s

import copy
import logging
from typing import Sequence

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph
from iree.turbine.kernel.wave.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.compiler.ir import Context, Location, Module
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.wave import LaunchableWave
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel.wave.utils import (
    get_default_compile_config,
    print_trace,
    run_test,
)
from iree.turbine.kernel.wave.constraints import MMAType


# Symbols
M, BLOCK_M, N, BLOCK_N, K, BLOCK_K, ADDRESS_SPACE, ELEMENTS_PER_LOAD, ELEMENTS_PER_STORE = (
    tkl.sym.M,
    tkl.sym.BLOCK_M,
    tkl.sym.N,
    tkl.sym.BLOCK_N,
    tkl.sym.K,
    tkl.sym.BLOCK_K,
    tkl.sym.ADDRESS_SPACE,
    tkl.sym.ELEMENTS_PER_LOAD,
    tkl.sym.ELEMENTS_PER_STORE,
)


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=kwargs["waves_per_block"]
            if "waves_per_block" in kwargs
            else (1, 1, 1),
            # One must always specify mma_type or vector_shapes.
            mma_type=MMAType.F32_16x16x16_F16,
        )
    ]
    return constraints


def harness_1d_global_mem(build_constraints_fun, kernel_fun, *args, **kwargs):
    constraints = build_constraints_fun(*args, **kwargs)
    with tk.gen.TestLaunchContext(
        kwargs["static_symbols"] if "static_symbols" in kwargs else {}
    ):
        lw = LaunchableWave(constraints, "kernel_fun", kernel_fun)

        trace: CapturedTrace = lw._trace()
        idxc: IndexingContext = IndexingContext.current()
        graph_passes = lw.build_initial_pass_pipeline(trace)
        for p in graph_passes:
            lw.try_apply_pass(p, trace)

        lw.infer_grid_shape(idxc)

        with Context() as context:
            mb, trace, exe, kernel_sig, entrypoint_name = lw.compile_to_mlir(
                trace, context, compile_config=get_default_compile_config(), **kwargs
            )
            print(mb.module_op)


def simple_mm_elem_wise_mul(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        # a_reg: tkw.Register[M, K, tkl.f16]
        a_reg = tkw.read(a, elements_per_thread=ELEMENTS_PER_LOAD)
        # b_reg: tkw.Register[N, K, tkl.f16]
        b_reg = tkw.read(b, elements_per_thread=ELEMENTS_PER_LOAD)
        # acc: tkw.Register[M, N, tkl.f32]
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    d_reg = tkw.read(d, elements_per_thread=ELEMENTS_PER_STORE)
    res = repeat * d_reg
    tkw.write(res, c, elements_per_thread=ELEMENTS_PER_STORE)


def static_config_256x256x8_mfma_16x16x16xf16():
    return {
        "static_symbols": {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ELEMENTS_PER_LOAD: 4,
            ELEMENTS_PER_STORE: 4,
            BLOCK_M: 64,
            BLOCK_N: 32,
            BLOCK_K: 32,
            M: 256,
            N: 256,
            K: 8,
        },
       #"vector_shapes": {M: 1},
        "waves_per_block": (2, 1, 1),
        # "dynamic_symbols": [M],
        "canonicalize": {True},
    }


@run_test
def static_correct_1():
    cfg = copy.deepcopy(static_config_256x256x8_mfma_16x16x16xf16())
    # CHECK-LABEL: static_correct_1
    #         CHECK: func.func @kernel_fun
    harness_1d_global_mem(build_block_constraints, simple_mm_elem_wise_mul, **cfg)
