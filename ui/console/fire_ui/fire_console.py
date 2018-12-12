import fire
from equations import membrane


class Membrane(object):

    def anim_graph_2d(self):
        membrane.main_2d()


if __name__ == '__main__':
    fire.Fire(Membrane)
