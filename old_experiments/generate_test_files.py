from .extract_params import ParamLoader
from .generate_random_filter import generate_random_directions
import os

def generate_test_file(base_folder, outfile, x_window, y_window, xpos, ypos):
    extension = "."+outfile.split(".")[-1]
    base_paraloader = ParamLoader(os.path.join(base_folder, "base_params"+extension))
    base_params = base_paraloader.get_params()
    x_dir_params = ParamLoader(os.path.join(base_folder, "x_dir"+extension)).get_params()
    y_dir_params = ParamLoader(os.path.join(base_folder, "y_dir"+extension)).get_params()

    x_loc = (xpos/x_window)*2-1
    y_loc = (ypos/y_window)*2-1
    alt_params = [param + dir1*x_loc + dir2*y_loc for dir1,dir2,param in zip(x_dir_params,y_dir_params,base_params)]
    for p, ap in zip(base_params, alt_params):
        assert p.shape == ap.shape, f"{p.shape},{ap.shape}"

    base_paraloader.set_params(alt_params)
    base_paraloader.save(outfile)

if __name__ == "__main__":
    generate_test_files("trained_agents/trpo/LunarLander-v2.pkl", "antbullettest/")
