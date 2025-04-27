from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.speculative_decoding import (
    get_speculative_sampling_kernel,
)


@run_test
def test_speculative_decoding():
    # Get the kernel and its hyperparameters
    kernel, hyperparams, _, _ = get_speculative_sampling_kernel(
        batch_size=64,
        num_speculative_tokens=3,
        num_draft_tokens=6,
        d=20,
        threshold_acc=1.0,
        threshold_single=1.0,
    )

    # Create the kernel with the hyperparameters
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        compile_to_mlir=True,
    )
    kernel = wave_compile(options, kernel)
    print(kernel.asm)
