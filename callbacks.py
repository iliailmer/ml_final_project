from catalyst.dl import Callback, CallbackOrder, State


class ValidationCallback(Callback):
    def __init__(self, ):
        super().__init__(CallbackOrder.Validation)

    def on_epoch_end(self, state: State):
        state.output['logits']
