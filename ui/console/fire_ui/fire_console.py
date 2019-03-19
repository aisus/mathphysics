import fire
from equations import analytic_solution


class Membrane(object):

    def anim_graph_2d(self):
        analytic_solution.animated_2d()


if __name__ == '__main__':
    fire.Fire(Membrane)
