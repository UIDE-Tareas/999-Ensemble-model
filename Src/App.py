"""Main application router for project commands."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Predictive maintenance ensemble model application."
    )

    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument(
        "-Train",
        action="store_true",
        help="Run the training workflow.",
    )
    command_group.add_argument(
        "-Predict",
        action="store_true",
        help="Run the prediction workflow.",
    )
    command_group.add_argument(
        "-Dashboard",
        action="store_true",
        help="Run the dashboard workflow.",
    )
    command_group.add_argument(
        "-Deps",
        action="store_true",
        help="Install project dependencies.",
    )

    args = parser.parse_args()

    if args.Train:
        from AppTrain import main as train_main

        train_main()
        return

    if args.Predict:
        from AppPredict import main as predict_main

        predict_main()
        return

    if args.Dashboard:
        from AppDashboard import main as dashboard_main

        dashboard_main()
        return

    if args.Deps:
        from Core.Utils import BASE_DEPENDENCIES
        from Core.Utils import InstallDeps

        InstallDeps(BASE_DEPENDENCIES)
        return


if __name__ == "__main__":
    main()
