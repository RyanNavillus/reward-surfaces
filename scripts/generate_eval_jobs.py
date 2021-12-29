import argparse
import warnings
from reward_surfaces.experiments import generate_eval_jobs


def main():
    parser = argparse.ArgumentParser(description='Generate jobs to evaluate directions/gradients/hessians')
    parser.add_argument('train_dir', type=str, help="Directory containing training checkpoints")
    parser.add_argument('out_dir', type=str, help="Directory to output evaluation info")
    parser.add_argument('--device', type=str, default='cuda', help="Device used for training ('cpu' or 'cuda')")
    parser.add_argument('--est-hesh', action='store_true', help="Estimate Hessian (Beta Feature)")
    parser.add_argument('--est-grad', action='store_true', help="Estimate Gradient (Beta Feature)")
    parser.add_argument('--calc-hesh', action='store_true', help="Calculate Hessian")
    parser.add_argument('--calc-grad', action='store_true', help="Calculate Gradient")
    parser.add_argument('--batch-grad', action='store_true', help="Calculate gradient by batching experience collection. Recommended for image observations")
    parser.add_argument('--num-steps', type=int, help="Number of steps used in evaluation")
    parser.add_argument('--num-episodes', type=int, help="Number of episodes used in evaluation")
    parser.add_argument('--checkpoint', type=str, help="Evaluate only a single checkpoint")

    args = parser.parse_args()

    if args.est_hesh or args.est_grad:
        warnings.warn("Hessian and gradient estimation are untested beta features.")

    generate_eval_jobs(
        args.train_dir,
        args.out_dir,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        est_hesh=args.est_hesh,
        calc_hesh=args.calc_hesh,
        calc_grad=args.calc_grad,
        batch_grad=args.batch_grad,
        est_grad=args.est_grad,
        device=args.device,
        checkpoint=args.checkpoint
    )


if __name__ == "__main__":
    main()
