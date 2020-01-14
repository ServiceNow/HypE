class Measure:
    def __init__(self):
        self.hit1  = {"raw": 0.0, "fil": 0.0}
        self.hit3  = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr   = {"raw": 0.0, "fil": 0.0}
        self.mr    = {"raw": 0.0, "fil": 0.0}

    def __add__(self, other):
        """
        Adding two measure objects
        """
        new_measure = Measure()
        settings = ["raw", "fil"]

        for rf in settings:
            new_measure.hit1[rf] = (self.hit1[rf] + other.hit1[rf])
            new_measure.hit3[rf] = (self.hit3[rf] + other.hit3[rf])
            new_measure.hit10[rf] = (self.hit10[rf] + other.hit10[rf])
            new_measure.mrr[rf] = (self.mrr[rf] + other.mrr[rf])
            new_measure.mr[rf] = (self.mr[rf] + other.mr[rf])
        return new_measure

    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil]  += rank
        self.mrr[raw_or_fil] += (1.0 / rank)

    def normalize(self, normalizer):
        if normalizer == 0:
            return
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil]  /= (normalizer)
            self.hit3[raw_or_fil]  /= (normalizer)
            self.hit10[raw_or_fil] /= (normalizer)
            self.mr[raw_or_fil]    /= (normalizer)
            self.mrr[raw_or_fil]   /= (normalizer)

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =",  self.hit1[raw_or_fil])
            print("\tHit@3 =",  self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =",     self.mr[raw_or_fil])
            print("\tMRR =",    self.mrr[raw_or_fil])
            print("")

