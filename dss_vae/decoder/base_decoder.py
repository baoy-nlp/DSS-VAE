class BaseDecoder(object):
    def __init__(self, name="BaseDecoder"):
        self.name = name
        print(name)

    def decode(self, **kwargs):
        """
        Using for greedy decoding for a given input, may corresponding with gold target,
        return the log_probability of decoder steps.
        """
        raise NotImplementedError

    def score(self, **kwargs):
        """
        Used for teacher-forcing training,
        return the log_probability of <input,output>.
        """
        raise NotImplementedError
