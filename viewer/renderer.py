from model.vanilla import VanillaGaussian


class ViewerRenderer:
    def __init__(
            self,
            model: VanillaGaussian,
    ):
        super().__init__()

        self.model = model

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        return self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
        )["render"]
        
        