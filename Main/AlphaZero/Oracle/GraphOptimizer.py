from Main import MachineSpecificSettings
if(MachineSpecificSettings.IS_UNIX_MACHINE):
    import tensorflow.contrib.tensorrt as tensorrt


def createOptimizedGraph(model, session, tf):
    outputs = [out.op.name for out in model.outputs]
    frozenGraph = _freeze_session(session, tf, output_names=outputs)
    # print("Model is now frozen...")
    # Possible precision modes: FP32, FP16, INT8
    optimizedGraph = tensorrt.create_inference_graph(frozenGraph, [out.op.name for out in model.outputs],
                                                     max_batch_size=MachineSpecificSettings.OPTIMIZED_GRAPH_MAX_BATCH,
                                                     precision_mode='FP32')

    return optimizedGraph, outputs


def _freeze_session(session, tf, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
