"""Fixed-size epochs carved out of a larger corpus.

Set `num_per_epoch` and one epoch becomes M items instead of the whole corpus.
This does NOT change how much training happens -- the budget is the number of
optimizer steps, set by the LR schedule. It changes how often you get a
validation point and a checkpoint, which is what makes a long run observable.

ROTATING PARTITION, NOT SAMPLING WITH REPLACEMENT
-------------------------------------------------
Epoch k takes the window [k*M, (k+1)*M) of a shuffled permutation of the
corpus, wrapping to a freshly shuffled permutation each time it runs off the
end. Every item is therefore seen exactly once per sweep.

The obvious alternative -- drawing M items at random every epoch -- leaves each
item unseen with probability (1 - 1/N)^D ~ e^(-D/N) after D draws, the
occupancy result. At D/N = 2 that is 13.5% of the corpus never seen at all, so
a with-replacement scheme needs D/N >= 4 before coverage stops being a concern.
The rotating partition reaches full coverage at D/N >= 1, which is worth roughly
a factor of four in wall-clock on a fixed-pair corpus like LRS2-2mix.

USAGE
-----
    train:  num_per_epoch: 8000   resample_per_epoch: true
    val:    num_per_epoch: 1000   resample_per_epoch: false

Validation must use `resample_per_epoch: false`: it then always returns the
same first M items, so the validation curve measures the model rather than
which subset the epoch happened to draw.

The trainer calls `set_epoch()` on the dataset each epoch (see
DDP_Trainer_Skeleton.run), mirroring how the DDP sampler is advanced. Leave
`num_per_epoch` unset for the original full-corpus behaviour.
"""

import numpy as np


class EpochSubsetMixin:
    def init_epoch_subset(self, corpus_size, num_per_epoch=None,
                          resample_per_epoch=True, seed=0):
        self.corpus_size = corpus_size
        self.num_per_epoch = num_per_epoch
        self.resample_per_epoch = resample_per_epoch
        self.subset_seed = seed if seed is not None else 0
        self._epoch = 0
        self._cached_epoch = None
        self._cached_indices = None

    @property
    def uses_epoch_subset(self):
        return getattr(self, 'num_per_epoch', None) is not None

    def set_epoch(self, epoch):
        """Advance the rotation. A no-op when num_per_epoch is unset."""
        self._epoch = int(epoch)

    def _permutation(self, sweep):
        rng = np.random.RandomState(self.subset_seed + int(sweep))
        return rng.permutation(self.corpus_size)

    def _epoch_indices(self):
        M, N = self.num_per_epoch, self.corpus_size
        if not self.resample_per_epoch:
            # Deterministic: always the same first M of one fixed shuffle.
            return self._permutation(0)[:M] if M < N else np.arange(N)

        if self._cached_epoch == self._epoch:
            return self._cached_indices

        start = (self._epoch * M) % N
        sweep = (self._epoch * M) // N
        out, pos, remaining = [], start, M
        while remaining > 0:
            perm = self._permutation(sweep)
            take = min(remaining, N - pos)
            out.append(perm[pos:pos + take])
            remaining -= take
            pos = 0
            sweep += 1
        indices = np.concatenate(out)

        self._cached_epoch, self._cached_indices = self._epoch, indices
        return indices

    def map_index(self, index):
        """Logical index within the epoch -> index into the corpus lists."""
        if not self.uses_epoch_subset:
            return index
        return int(self._epoch_indices()[index % self.num_per_epoch])

    def epoch_length(self, full_length):
        """How many items this epoch holds."""
        if not self.uses_epoch_subset:
            return full_length
        return min(self.num_per_epoch, self.corpus_size)
