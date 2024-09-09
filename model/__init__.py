def get_model(cfg, scene):
    if cfg.model.name == "vanilla":
        from .vanilla import VanillaGaussian
        return VanillaGaussian(cfg, scene)
    elif cfg.model.name == "unc_2d_unet":
        from .unc_2d_unet import Unc2DUnet
        return Unc2DUnet(cfg, scene)
    elif cfg.model.name == "deform":
        from .deform import DeformGaussian
        return DeformGaussian(cfg, scene)
    elif cfg.model.name == "deform_static":
        from .deform_static import DeformStaticGaussian
        return DeformStaticGaussian(cfg, scene)
    elif cfg.model.name == "gaussian_grouping":
        from .gaussian_grouping import GaussianGrouping
        return GaussianGrouping(cfg, scene)
    else:
        raise Exception(f"ERROR: Model {cfg.model.name} not implemented. ")