class GlobalNames(object):
    fm = None

    @staticmethod
    def set_judge(fm):
        GlobalNames.fm = fm

    @staticmethod
    def get_fm():
        return GlobalNames.fm
