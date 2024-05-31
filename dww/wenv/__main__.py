from dww.wenv.config import WaterWorldConfig, raise_compliant
from dww.wenv import registry
from foo.utils import dummy_render, dummy_measure
from pprint import pprint

# Add a custom wenv config to the registry
registry.update({'my_env': WaterWorldConfig(n_pursuers=8, n_coop=4, local_ratio=0.8),
                 'my_env2': WaterWorldConfig(n_pursuers=8,
                                             n_coop=6,
                                             local_ratio=0.8)})

if __name__ == "__main__":

    NAME = 'my_env2'
    env, conf, sha = registry(NAME, render_mode='human', FPS=30)  # get wenv overriding render_mode
    # the next helper should be called for serious exps relative
    # to the final paper to check that indeed all parameters that should
    # be fixed, indeed are.
    raise_compliant(conf)

    print(NAME, sha, env, type(env))
    pprint(conf.manipulable_values())
    pprint(conf)

    # now we want to show the rendering of the game

    DUMMY_RENDER = True
    if DUMMY_RENDER:
        print("Rendering wenv")
        its_per_sec = dummy_render(env, conf, 150)  # run rendering for 300 frames
        print(f"Rendering: {its_per_sec} [it/sec]")

    # measure wenv loop speed
    DUMMY_MEASURE = True
    if DUMMY_MEASURE:
        print("Measure wenv it/sec without rendering")
        env2, _, _ = registry(NAME)  # don't set human mode on this time
        its_per_sec = dummy_measure(env2, conf, 1000)
        print(f"{its_per_sec} [it/sec]")
        print(f"{its_per_sec / conf.n_pursuers} [ag*it/sec]")
