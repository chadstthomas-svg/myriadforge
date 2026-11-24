"""OpenVINO 2022.3 compilation for synchronous Myriad X callers.

This module is importable in the isolated legacy Python environment.
"""


def compile_myriad_x(core, ir, shaves):
    from openvino.runtime import Dimension

    model = core.read_model(str(ir))
    batch_one = {}
    for index, port in enumerate(model.inputs):
        shape = port.get_partial_shape()
        if (shape.rank.is_static and len(shape) and shape[0].is_dynamic
                and shape[0].compatible(Dimension(1))
                and all(dim.is_static for dim in list(shape)[1:])):
            shape[0] = Dimension(1)
            batch_one[index] = shape
    if batch_one:
        model.reshape(batch_one)

    # The synchronous callers use one stream, so the allocation is per model.
    return core.compile_model(model, "MYRIAD", {
        "MYRIAD_THROUGHPUT_STREAMS": 1,
        "MYRIAD_NUMBER_OF_SHAVES": shaves,
        "MYRIAD_NUMBER_OF_CMX_SLICES": shaves,
    })
