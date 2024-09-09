import argparse
from arguments import ModelParams, PipelineParams, get_combined_args


def to_gs_args(args, stride=500):
    parser = argparse.ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    cmd_list = [
        "-s", str(args.data),
        "-m", str(args.input),
        "--dim_extra", str(args.feat_dim),
        "--camera_name", "rgb",
        "--data_device", "cpu",
        "--stride", str(stride),
    ]

    # Override the rendering resolution if specified in this script. 
    # if args.render_resolution is not None:
    #     cmd_list += ["--resolution", str(args.render_resolution)]
    
    args_gs = get_combined_args(parser, cmd_list)

    args_scene = model.extract(args_gs)
    args_pipe = pipeline.extract(args_gs)

    return args_scene, args_pipe